"模型的定义"

# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    实现基于正弦和余弦函数的位置编码，用于为输入序列中的每个位置提供唯一的表示。

    工作流程:
    1. 初始化时，根据模型维度 `dim_model` 计算出频率的倒数 `inv_freq`，用于构造正弦/余弦函数
    2. 在前向传播中，根据输入的序列长度生成位置索引
    3. 使用外积计算每个位置和频率的乘积
    4. 分别对这些乘积应用正弦和余弦函数，并与输入张量进行相加，增强位置信息

    Args:
        dim_model: 模型的隐藏维度，必须为偶数。

    Returns:
        一个形状为 (batch_size, seq_len, dim_model) 的位置编码张量，与输入 `x` 的数据类型相同。

    Examples:
        >>> pe = PositionalEncoding(512)
        >>> x = torch.randn(32, 128, 512)  # batch_size=32, seq_len=128, dim_model=512
        >>> x = pe(x)  # 返回形状为 (32, 128, 512) 的经过位置编码的张量
    """

    def __init__(self, dim_model: int):
        super().__init__()

        # 只对偶数维度（从0开始每隔两位）计算
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))

        # 注册为buffer，意味着它不是参数，不会被训练，且不随模型保存
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)

        # 创建位置索引张量（长度为序列长度），并转换为与 inv_freq 相同的数据类型
        positions = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)

        # 计算外积: positions[i] * inv_freq[j]
        sinusoid_input = torch.einsum("i,j->ij", positions, self.inv_freq)

        # 最后一维合并为 dim_model 维度
        positional_embedding = torch.cat([sinusoid_input.sin(), sinusoid_input.cos()], dim=-1)

        # 增加 batch 维度，并确保返回类型与输入相同
        return x + positional_embedding.unsqueeze(0).type_as(x)


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
        self.g = nn.Parameter(torch.ones(1, device=device) * (dim ** 0.5))  # 可学习的缩放因子，初始化为dim的平方根
        self.eps = eps  # 避免除零错误的小常数

    def forward(self, x):
        """
        前向传播计算。通过L2范数对输入张量 `x` 进行缩放归一化。

        Args:
            x: 输入的张量，形状可以为任意维度。

        Returns:
            返回经过缩放归一化后的张量，形状与输入相同。
        """
        norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)  # 计算L2范数并防止为零
        return self.g * x / norm  # 对输入张量进行缩放归一化


class MultiqueryAttention(nn.Module):
    """
    多查询注意力 (Multi-Query Attention)。
    与标准多头注意力不同，多查询注意力共享键和值的头，
    仅查询保持多头，可显著减少计算量和内存占用。

    1. 键/值投影合并为单头（`dim_head`维），查询保持多头（`num_heads * dim_head`维）
    2. 计算时通过广播机制将键/值复制到与查询相同的头数
    3. 使用PyTorch原生`scaled_dot_product_attention`实现高效注意力计算

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

        # 处理注意力掩码
        if padding_mask is None:
            # 使用内置的注意力掩码
            attn_mask = None
            use_builtin_causal = True
        else:
            # 不使用内置的因果注意力（因为不能同时使用`attn_mask`和`is_causal`，需要自定义掩码）
            use_builtin_causal = False

            # 创建因果掩码 [seq_len, seq_len]
            causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(diagonal=1)

            # 生成注意力掩码 (同时考虑因果和padding)
            attn_mask = causal_mask.unsqueeze(0) | padding_mask.view(batch_size, 1, 1, seq_len) | padding_mask.view(batch_size, 1, seq_len, 1)

            # 将布尔掩码转换为浮点掩码
            attn_mask = torch.where(attn_mask, -torch.inf, 0.)

        # 计算缩放点积注意力
        attn_output = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_mask,
            dropout_p=(self.dropout_rate if self.training else 0.),
            is_causal=use_builtin_causal
        )

        # 合并多头输出 (batch, seq_len, dim_model)
        return attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)


class MidiNetLayer(nn.Module):
    """
    MidiNetLayer 是一个神经网络层，结合了注意力机制和前馈网络，常用于序列数据建模。该层包含以下组件：
    - 使用 FlashAttention 进行高效的自注意力计算。
    - 线性前馈网络（带有 GELU 激活函数）。
    - 使用 ScaleNorm 替代传统的 LayerNorm 进行归一化，以提升计算效率。
    - Dropout 层，避免过拟合。

    Args:
        num_heads: 注意力头的数量。
        dim_head: 每个注意力头的维度。
        dim_feedforward: 前馈网络的隐藏层维度。
        dropout: Dropout 概率，用于防止过拟合。
        device: 模型的设备。
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

        # 多查询注意力
        self.attention = MultiqueryAttention(dim_head, num_heads, dropout=dropout, device=device)

        # 前馈网络部分: 线性 -> GELU 激活 -> 线性
        self.feedforward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward, device=device),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim_feedforward, dim_model, device=device)
        )

        # 使用 ScaleNorm 归一化，替代 LayerNorm 以提升效率和性能
        self.norm1 = ScaleNorm(dim_model, device=device)
        self.norm2 = ScaleNorm(dim_model, device=device)

        # 添加 Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        for param in self.feedforward.parameters():
            if isinstance(param, nn.Linear):
                torch.nn.init.xavier_uniform_(param.weight)
                torch.nn.init.zeros_(param.bias)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        执行前向传播操作，输入通过注意力模块和前馈网络后输出。

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim_model)。
            padding_mask: 可选的 padding 掩码，用于注意力机制中忽略填充部分。

        Returns:
            输出张量，形状与输入相同。
        """
        # 执行自注意力计算，添加残差连接，并通过 Dropout 防止过拟合
        x = self.dropout(x + self.attention(self.norm1(x), padding_mask=padding_mask))

        # 执行前馈网络计算，添加残差连接，并通过 Dropout 防止过拟合
        x = self.dropout(x + self.feedforward(self.norm2(x)))

        return x


class MidiNet(nn.Module):
    """
    Midi 音乐生成模型。

    该模型通过嵌入、位置编码、多个堆叠的 MidiNetLayer（注意力 + 前馈网络）、输出层，
    将输入的 token 序列转换为下一个 token 的概率分布，可用于 Midi 音符的生成任务。

    - 使用共享权重的嵌入层和输出层，减少参数数量并提高模型效果。
    - 多层堆叠的 Transformer 样式结构，支持捕捉复杂的时间依赖关系。
    - 使用可选的 Dropout 机制进行正则化。

    Args:
        vocab_size: 词汇表大小。
        num_heads: 注意力头数量。
        dim_head: 每个注意力头的维度。
        dim_feedforward: 前馈网络的隐藏层维度。
        num_layers: Transformer 层的数量。
        dropout: Dropout 概率。
        device: 模型所在设备。
    """

    def __init__(
        self,
        vocab_size: int,
        num_heads: int,
        dim_head: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
        device: torch.device = None
    ):
        super().__init__()
        self.dim_model = dim_head * num_heads  # 模型总维度

        # 将 token 映射为向量
        self.embedding = nn.Embedding(vocab_size, self.dim_model, device=device)

        # 位置编码
        self.positional_encoding = PositionalEncoding(self.dim_model)

        # 堆叠多个 MidiNetLayer 层
        layer = MidiNetLayer(num_heads, dim_head, dim_feedforward, dropout=dropout, device=device)
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))

        # 将模型输出映射回 vocab 空间
        self.output_layer = nn.Linear(self.dim_model, vocab_size, device=device)

        # 添加 Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 嵌入层与输出层共享权重，提升效率与性能
        self.output_layer.weight = self.embedding.weight

        # 初始化权重
        torch.nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None):
        # 将 token 转为向量，并乘以 sqrt(dim_model) 进行缩放
        x = self.dropout(self.embedding(x) * math.sqrt(self.dim_model))

        # 添加位置编码
        x = self.positional_encoding(x)

        # 逐层应用 Transformer 结构
        for layer in self.layers:
            # 进入单层 Transformer 结构
            x = layer(x, padding_mask)

        # 映射回词汇表空间，得到每个位置的预测分布
        logits = self.output_layer(x)
        return logits
