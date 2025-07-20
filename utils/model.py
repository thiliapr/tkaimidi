"""
这个模块实现了 MidiNet 模型的核心结构，包括位置编码、多查询注意力、前馈网络等组件。
MidiNet 是一个基于 Transformer 的模型，专门用于 MIDI 音乐生成任务。
"""

# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
import copy
from typing import NamedTuple, Optional
import torch
from torch import nn
from torch.nn import functional as F


class MidiNetConfig(NamedTuple):
    """
    MidiNet 的配置。

    Attributes:
        vocab_size: 词汇表大小。
        num_heads: 注意力头数量。
        dim_head: 每个注意力头的维度。
        dim_feedforward: 前馈网络的隐藏层维度。
        num_layers: Transformer 层的数量。
    """
    vocab_size: int
    num_heads: int
    dim_head: int
    dim_feedforward: int
    num_layers: int


class ScaleNorm(nn.Module):
    """
    实现了一个缩放归一化的模块。该模块根据输入的张量 `x` 计算其L2范数，并用可学习的缩放因子 `g` 对其进行缩放归一化。用于增强模型的稳定性和学习能力。

    工作流程如下：
        1. 输入张量 `x` 会计算其在最后一个维度上的L2范数。
        2. 通过一个可学习的参数 `g` 对输入进行缩放。
        3. 输出结果是经过缩放的归一化张量。

    Args:
        dim: 用于初始化缩放因子 `g` 的维度。
        eps: 防止除零的常数（默认为1e-5）。

    Returns:
        返回缩放归一化后的张量。

    Examples:
        # 示例代码：
        scale_norm = ScaleNorm(dim=256)
        x = torch.randn(10, 256)  # 假设x的维度是[10, 256]
        output = scale_norm(x)  # 返回经过缩放归一化后的张量
    """

    def __init__(self, dim: int, eps: float = 1e-5, device: Optional[torch.device] = None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, device=device) * (dim ** 0.5))  # 可学习的缩放因子，初始化为dim的平方根
        self.eps = eps  # 避免除零错误的小常数

    def forward(self, x):
        norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)  # 计算L2范数并防止为零
        return self.scale * x / norm  # 对输入张量进行缩放归一化


class MultiqueryAttention(nn.Module):
    """
    多查询注意力 (Multi-Query Attention)。
    与标准多头注意力不同，多查询注意力共享键和值的头，
    仅查询保持多头，可显著减少计算量和内存占用。

    1. 键/值投影合并为单头（`dim_head`维），查询保持多头（`num_heads * dim_head`维）
    2. 计算时通过广播机制将键/值复制到与查询相同的头数
    3. 使用PyTorch原生`scaled_dot_product_attention`实现高效注意力计算
    4. 支持 RoPE（旋转位置编码）对 Q/K 应用

    Args:
        dim_head: 每个注意力头的维度
        num_heads: 注意力头的数量
        dropout: 注意力权重dropout概率，默认为0
        device: 模型参数所在的设备

    Examples:
        >>> attention = MultiqueryAttention(dim_head=64, num_heads=8)
        >>> x = torch.randn(32, 100, 512)  # (batch, seq_len, dim)
        >>> output = attention(x)
    """

    def __init__(self, dim_head: int, num_heads: int, dropout: float = 0., device: torch.device = None):
        super().__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dropout_rate = dropout
        dim_model = dim_head * num_heads  # 总模型维度 = 头数 * 每头维度

        # 查询、键值投影矩阵 (合并计算效率更高)
        # 输出维度: Queries (dim_model) + Keys (dim_head) + Values (dim_head)
        self.qkv_proj = nn.Linear(dim_model, dim_model + dim_head * 2, device=device)

        # 输出投影矩阵，将多头输出合并回原始维度
        self.out_proj = nn.Linear(dim_model, dim_model, device=device)

        # RoPE 旋转频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        # 使用 Xavier 均匀分布初始化查询、键值投影权重
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def apply_rope(self, x: torch.Tensor, seq_len: int):
        """
        对输入张量 x 应用旋转位置编码（Rotary Positional Embedding, RoPE）。

        Args:
            x: 输入张量，形状为 [batch, heads, seq_len, dim_head]，表示多头注意力中的特征表示。
            seq_len: 序列长度，通常等于 x 的第三维长度。

        Returns:
            应用 RoPE 后的张量，形状与输入 x 相同。

        详细说明:
            - 仅对最后一维（dim_head）进行旋转位置编码处理。
            - 首先根据 inv_freq（RoPE 的频率参数）生成正弦和余弦位置编码 emb，形状为 [seq_len, dim_head]。
            - emb 通过 unsqueeze 扩展为 [1, 1, seq_len, dim_head]，以便与输入 x 广播相乘。
            - 将 x 的最后一维拆分为偶数索引（x1）和奇数索引（x2），分别与 emb 的前后半部分进行组合，实现旋转编码。
            - 最终返回与输入形状一致的编码后张量。

        注意事项:
            - self.rope_inv_freq 应为 [dim_head // 2] 形状的频率张量。
            - self.dim_head 应为最后一维的特征维度，且为偶数。
        """
        # x: [batch, heads, seq_len, dim_head]
        # 只对最后一维（特征维度 dim_head）做 RoPE 旋转位置编码
        device = x.device
        inv_freq = self.rope_inv_freq  # [dim_head // 2]，每个偶数维度的旋转频率

        # 生成位置索引 t: [seq_len]
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)

        # 计算每个位置和每个频率的乘积，得到 [seq_len, dim_head//2]
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # 拼接正弦和余弦编码，得到 [seq_len, dim_head]
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)

        # 扩展 emb 形状为 [1, 1, seq_len, dim_head]，方便与 x 广播相乘
        emb = emb.unsqueeze(0).unsqueeze(0)

        # 拆分 x 的最后一维为偶数索引（x1）和奇数索引（x2），形状均为 [..., dim_head//2]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rope = torch.empty_like(x)

        # 偶数位置：x1 * cos - x2 * sin
        x_rope[..., ::2] = x1 * emb[..., :self.dim_head//2] - x2 * emb[..., self.dim_head//2:]

        # 奇数位置：x1 * sin + x2 * cos
        x_rope[..., 1::2] = x1 * emb[..., self.dim_head//2:] + x2 * emb[..., :self.dim_head//2]

        # 返回应用 RoPE 后的张量，形状与输入一致
        return x_rope

    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 计算查询、键值投影 [batch, seq_len, dim_model + 2 * dim_head]
        qkv = self.qkv_proj(x)

        # 分割查询、键和值 (注意顺序: Q -> K -> V)
        queries, keys, values = qkv.split([self.dim_head * self.num_heads, self.dim_head, self.dim_head], dim=-1)

        # 调整查询形状为 [batch, seq_len, num_heads, dim_head]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.dim_head)

        # 将键和值从单头扩展到多头 [batch, seq_len, 1, dim_head] -> [batch, seq_len, num_heads, dim_head]
        keys = keys.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        values = values.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        # 重排维度为 PyTorch 注意力要求的形状 [batch, heads, seq_len, dim_head]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 应用 RoPE 到 Q/K
        queries = self.apply_rope(queries, seq_len)
        keys = self.apply_rope(keys, seq_len)

        # 处理注意力掩码
        if padding_mask is None:
            attn_mask = None
            use_builtin_causal = True
        else:
            use_builtin_causal = False
            causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(diagonal=1)
            attn_mask = (
                causal_mask.unsqueeze(0)
                | padding_mask.unsqueeze(1)
                | padding_mask.unsqueeze(2)
            )
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = torch.where(attn_mask, -torch.inf, 0.)

        # 计算缩放点积注意力
        attn_output = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_mask,
            dropout_p=(self.dropout_rate if self.training else 0.),
            is_causal=use_builtin_causal
        )

        # 合并多头输出 (batch, seq_len, dim_model)
        return self.out_proj(attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1))


class MidiNetLayer(nn.Module):
    """
    MidiNet 的单层 Transformer 结构。
    包含多查询注意力和前馈网络部分，使用残差连接和缩放归一化。
    工作流程如下:
        1. 输入张量通过多查询注意力模块，计算注意力输出。
        2. 将注意力输出与输入张量相加，形成残差连接。
        3. 通过前馈网络部分，进一步处理注意力输出。
        4. 将前馈网络输出与注意力输出相加，形成最终输出。
    
    Args:
        num_heads: 注意力头的数量。
        dim_head: 每个注意力头的维度。
        dim_feedforward: 前馈网络的隐藏层维度。
        dropout: Dropout 概率。
        device: 模型参数所在的设备。
    
    Examples:
        >>> layer = MidiNetLayer(num_heads=8, dim_head=64, dim_feedforward=2048)
        >>> x = torch.randn(32, 100, 512)  # batch_size=32, seq_len=100, dim_model=512
        >>> output = layer(x)  # 返回形状为 (32, 100, 512) 的张量
    """

    def __init__(
        self,
        num_heads: int,
        dim_head: int,
        dim_feedforward: int,
        dropout: float = 0.,
        device: torch.device = None
    ):
        super().__init__()
        dim_model = dim_head * num_heads  # 模型总维度

        # 多查询注意力: 归一化 -> 多查询注意力
        # 使用 ScaleNorm 进行归一化，MultiqueryAttention 实现多查询注意力机制
        self.attention = MultiqueryAttention(dim_head, num_heads, dropout=dropout, device=device)

        # 前馈网络部分: 归一化 -> 线性 -> GELU 激活 -> 线性
        self.feedforward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward, device=device),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim_feedforward, dim_model, device=device)
        )

        # ScaleNorm 用于对注意力输出进行归一化，增强模型稳定性
        self.attention_norm = ScaleNorm(dim_model, device=device)
        self.feedforward_norm = ScaleNorm(dim_model, device=device)

        # 注意力和前馈网络的缩放因子
        self.attention_scale = nn.Parameter(torch.zeros(1, device=device))
        self.feedforward_scale = nn.Parameter(torch.zeros(1, device=device))

        # 添加 Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        for module in self.feedforward.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 对输入张量进行多查询注意力和前馈网络处理
        x = x + self.dropout(self.attention(self.attention_norm(x), padding_mask=padding_mask) * self.attention_scale)
        x = x + self.dropout(self.feedforward(self.feedforward_norm(x)) * self.feedforward_scale)

        # 返回最终输出张量
        return x


class MidiNet(nn.Module):
    """
    Midi 音乐生成模型。

    该模型通过嵌入、位置编码、多个堆叠的 MidiNetLayer（注意力 + 前馈网络）、输出层，
    将输入的 token 序列转换为下一个 token 的概率分布，可用于 Midi 音符的生成任务。

    - 使用共享权重的嵌入层和输出层，减少参数数量并提高模型效果。
    - 多层堆叠的 Transformer 样式结构，支持捕捉复杂的时间依赖关系。
    - 使用可选的 Dropout 机制进行正则化。
    - RoPE（旋转位置编码）可选应用于注意力计算，增强模型对序列位置的感知。

    Args:
        config: 模型的配置。
        dropout: Dropout 概率。
        device: 模型所在设备。
    """

    def __init__(
        self,
        config: MidiNetConfig,
        dropout: float = 0.1,
        device: torch.device = None
    ):
        super().__init__()
        self.dim_model = config.dim_head * config.num_heads  # 模型总维度

        # 将 token 映射为向量
        self.embedding = nn.Embedding(config.vocab_size, self.dim_model, device=device)

        # 堆叠多个 MidiNetLayer 层
        layer = MidiNetLayer(config.num_heads, config.dim_head, config.dim_feedforward, dropout=dropout, device=device)
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(config.num_layers))

        # 将模型输出映射回 vocab 空间
        self.output_layer = nn.Linear(self.dim_model, config.vocab_size, device=device)

        # 添加 Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None):
        # 将 token 转为向量，并乘以 sqrt(dim_model) 进行缩放
        x = self.embedding(x) * math.sqrt(self.dim_model)

        # 应用 Dropout
        x = self.dropout(x)

        # 逐层应用 Transformer 结构
        for layer in self.layers:
            # 进入单层 Transformer 结构
            x = layer(x, padding_mask)

        # 映射回词汇表空间，得到每个位置的预测分布
        logits = self.output_layer(x)
        return logits
