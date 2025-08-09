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

AttentionKVCache = tuple[torch.Tensor, torch.Tensor]
NetKVCache = list[AttentionKVCache]


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

    def __init__(self, dim_head: int, num_heads: int, dropout: float = 0., device: Optional[torch.device] = None):
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
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_head, 2, device=device) / dim_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # freqs_cis 缓存
        self.freqs_cis_cache: torch.Tensor
        self.register_buffer("freqs_cis_cache", torch.empty(0, dim_head // 2, device=device), persistent=False)

        # 使用 Xavier 均匀分布初始化查询、键值投影权重
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def apply_rope(self, x: torch.Tensor, cache_start_idx: int = 0) -> torch.Tensor:
        """
        应用旋转位置编码(RoPE)到输入张量。

        旋转位置编码通过复数乘法实现位置信息的注入：
        1. 将输入张量的最后两个维度视为复数对(实部和虚部)
        2. 生成与位置相关的旋转复数向量
        3. 通过复数乘法实现旋转操作
        4. 将旋转后的复数转换回实数表示

        该方法支持增量计算：
        - 维护旋转频率缓存(freqs_cis_cache)避免重复计算
        - 当序列长度超过缓存大小时自动扩展缓存
        - 支持从指定位置开始应用旋转编码

        Args:
            x: 输入张量，形状为 [batch, num_heads, seq_len, dim_head]
            cache_start_idx: 在旋转频率缓存中开始使用的位置索引
                - 用于支持增量解码场景，所以不考虑多 batch 情况
                - 例如：cache_start_idx=10 表示从缓存的第10个位置开始使用旋转频率
                - 默认值0表示从头开始使用旋转频率缓存

        Returns:
            应用旋转位置编码后的张量，形状与输入相同
        """
        # 计算需要应用RoPE的序列长度
        required_seq_len = x.size(2)

        # 检查并更新旋转频率缓存
        current_cache_len = self.freqs_cis_cache.size(0)
        if current_cache_len < cache_start_idx + required_seq_len:
            # 生成缺失位置的时间索引
            new_positions = torch.arange(
                current_cache_len,
                cache_start_idx + required_seq_len,
                device=self.inv_freq.device
            )
            # 计算新位置的旋转频率
            new_freqs = torch.outer(new_positions, self.inv_freq)
            # 转换为复数形式 (cosθ + i·sinθ)
            new_cis = torch.polar(torch.ones_like(new_freqs), new_freqs)
            # 更新缓存
            self.freqs_cis_cache = torch.cat([self.freqs_cis_cache, new_cis], dim=0)

        # 获取当前序列所需的旋转频率
        freqs_cis = self.freqs_cis_cache[cache_start_idx:cache_start_idx + required_seq_len]

        # 将最后维度重塑为复数对 (..., dim_head//2, 2)
        complex_shape = x.shape[:-1] + (-1, 2)
        complex_pairs = x.float().reshape(complex_shape)

        # 转换为复数张量
        complex_tensor = torch.view_as_complex(complex_pairs)

        # 调整旋转频率形状以匹配输入 (添加批量和头维度)
        freqs_cis = freqs_cis.view(1, 1, -1, freqs_cis.shape[-1])

        # 应用旋转（复数乘法）
        rotated_complex = complex_tensor * freqs_cis

        # 转换回实数表示
        rotated_real = torch.view_as_real(rotated_complex)

        # 展平最后两个维度 (..., dim_head//2, 2) -> (..., dim_head)
        rotated_output = rotated_real.flatten(-2).to(dtype=x.dtype)
        return rotated_output

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        kv_cache: Optional[AttentionKVCache] = None
    ) -> tuple[torch.Tensor, AttentionKVCache]:
        batch_size, seq_len, _ = x.shape

        if kv_cache is not None and padding_mask is not None:
            raise RuntimeError("padding_mask 和 kv_cache 不能同时使用。padding_mask 用于训练阶段的批次填充处理，kv_cache 用于推理阶段的增量解码。")

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
        if kv_cache is not None:
            cache_len = kv_cache[0].size(2)
            queries = self.apply_rope(queries, cache_start_idx=cache_len)
            keys = self.apply_rope(keys, cache_start_idx=cache_len)

            # 应用 KV Cache
            keys = torch.cat([kv_cache[0], keys], dim=2)
            values = torch.cat([kv_cache[1], values], dim=2)
        else:
            queries = self.apply_rope(queries)
            keys = self.apply_rope(keys)

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
        return self.out_proj(attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)), (keys, values)


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

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[AttentionKVCache] = None
    ) -> tuple[torch.Tensor, AttentionKVCache]:
        # 注意力模块
        attn_output, kv_cache = self.attention(self.attention_norm(x), padding_mask=padding_mask, kv_cache=kv_cache)
        x = x + self.dropout(attn_output * self.attention_scale)

        # 前馈网络
        x = x + self.dropout(self.feedforward(self.feedforward_norm(x)) * self.feedforward_scale)

        # 返回最终输出张量
        return x, kv_cache


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

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        kv_cache: Optional[NetKVCache] = None
    ) -> tuple[torch.Tensor, NetKVCache]:
        # 将 token 转为向量，并乘以 sqrt(dim_model) 进行缩放
        x = self.embedding(x) * math.sqrt(self.dim_model)

        # 应用 Dropout
        x = self.dropout(x)

        # 逐层应用 Transformer 结构
        layers_kv_cache = []
        for layer_idx, layer in enumerate(self.layers):
            if kv_cache:
                x, layer_kv_cache = layer(x, padding_mask, kv_cache[layer_idx])
            else:
                x, layer_kv_cache = layer(x, padding_mask)

            layers_kv_cache.append(layer_kv_cache)

        # 映射回词汇表空间，得到每个位置的预测分布
        logits = self.output_layer(x)
        return logits, layers_kv_cache
