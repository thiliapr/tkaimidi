"模型的定义"

# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import math
import torch
from torch import nn
from typing import Optional

# 在非 Jupyter 环境下导入注意力库
if "get_ipython" not in globals():
    from attention import FlashAttention


class PositionalEncoding(nn.Module):
    """
    实现基于正弦和余弦函数的位置编码，用于为输入序列中的每个位置提供唯一的表示。

    工作流程:
    1. 初始化时，根据模型维度 `model_dim` 计算出频率的倒数 `inv_freq`，用于构造正弦/余弦函数
    2. 在前向传播中，根据输入的序列长度生成位置索引
    3. 使用外积计算每个位置和频率的乘积
    4. 分别对这些乘积应用正弦和余弦函数，并与输入张量进行相加，增强位置信息

    Args:
        model_dim: 模型的隐藏维度，必须为偶数。

    Returns:
        一个形状为 (batch_size, seq_len, model_dim) 的位置编码张量，与输入 `x` 的数据类型相同。

    Examples:
        >>> pe = PositionalEncoding(512)
        >>> x = torch.randn(32, 128, 512)  # batch_size=32, seq_len=128, model_dim=512
        >>> x = pe(x)  # 返回形状为 (32, 128, 512) 的经过位置编码的张量
    """

    def __init__(self, model_dim: int):
        super().__init__()

        # 只对偶数维度（从0开始每隔两位）计算
        inv_freq = 1.0 / (10000 ** (torch.arange(0, model_dim, 2).float() / model_dim))

        # 注册为buffer，意味着它不是参数，不会被训练，且不随模型保存
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)

        # 创建位置索引张量（长度为序列长度），并转换为与 inv_freq 相同的数据类型
        positions = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)

        # 计算外积: positions[i] * inv_freq[j]
        sinusoid_input = torch.einsum("i,j->ij", positions, self.inv_freq)

        # 最后一维合并为 model_dim 维度
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


class MidiNetLayer(nn.Module):
    """
    MidiNetLayer 是一个神经网络层，结合了注意力机制和前馈网络，常用于序列数据建模。该层包含以下组件：
    - 使用 FlashAttention 进行高效的自注意力计算。
    - 线性前馈网络（带有 GELU 激活函数）。
    - 使用 ScaleNorm 替代传统的 LayerNorm 进行归一化，以提升计算效率。
    - Dropout 层，避免过拟合。

    Args:
        num_heads: 注意力头的数量。
        head_dim: 每个注意力头的维度。
        feedforward_dim: 前馈网络的隐藏层维度。
        dropout: Dropout 概率，用于防止过拟合。
        device: 模型的设备（如 CPU 或 GPU。
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        feedforward_dim: int,
        dropout: float = 0.,
        device: torch.device = None
    ):
        super().__init__()
        model_dim = num_heads * head_dim  # 模型总维度

        # 使用 FlashAttention 实现高效的多头注意力
        self.attention = FlashAttention(dim=model_dim, heads=num_heads, dim_head=head_dim, causal=True).to(device)

        # 前馈网络部分: 线性 -> GELU 激活 -> 线性
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim, device=device),
            nn.GELU(),
            nn.Linear(feedforward_dim, model_dim, device=device)
        )

        # 使用 ScaleNorm 归一化，替代 LayerNorm 以提升效率和性能
        self.norm1 = ScaleNorm(model_dim)
        self.norm2 = ScaleNorm(model_dim)

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
            x: 输入张量，形状为 (batch_size, seq_len, model_dim)。
            padding_mask: 可选的 padding 掩码，用于注意力机制中忽略填充部分。

        Returns:
            输出张量，形状与输入相同。
        """
        # 执行自注意力计算，添加残差连接，并通过 Dropout 防止过拟合
        x = self.dropout(x + self.attention(self.norm1(x), mask=padding_mask))

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
        head_dim: 每个注意力头的维度。
        feedforward_dim: 前馈网络的隐藏层维度。
        num_layers: Transformer 层的数量。
        dropout: Dropout 概率。
        device: 模型所在设备。
    """

    def __init__(
        self,
        vocab_size: int,
        num_heads: int,
        head_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        device: torch.device = None
    ):
        super().__init__()
        self.model_dim = head_dim * num_heads  # 模型总维度

        # 将 token 映射为向量
        self.embedding = nn.Embedding(vocab_size, self.model_dim, device=device)

        # 位置编码
        self.positional_encoding = PositionalEncoding(self.model_dim)

        # 堆叠多个 MidiNetLayer 层
        self.layers = nn.ModuleList(
            MidiNetLayer(num_heads, head_dim, feedforward_dim, dropout=dropout, device=device)
            for _ in range(num_layers)
        )

        # 将模型输出映射回 vocab 空间
        self.output_layer = nn.Linear(self.model_dim, vocab_size, device=device)

        # 添加 Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 嵌入层与输出层共享权重，提升效率与性能
        self.output_layer.weight = self.embedding.weight

        # 初始化权重
        torch.nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None):
        # 将 token 转为向量，并乘以 sqrt(model_dim) 进行缩放
        x = self.dropout(self.embedding(x) * math.sqrt(self.model_dim))

        # 添加位置编码
        x = self.positional_encoding(x)

        # 逐层应用 Transformer 结构
        for layer in self.layers:
            # 进入单层 Transformer 结构
            x = layer(x)

        # 映射回词汇表空间，得到每个位置的预测分布
        logits = self.output_layer(x)
        return logits
