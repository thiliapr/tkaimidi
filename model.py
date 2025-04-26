"模型的定义"

# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEmbedding(nn.Module):
    """
    实现带有自注意力机制和位置感知前馈网络的位置嵌入模块。

    该模块包含两个主要部分:
    1. 多头自注意力机制: 用于捕捉序列中不同位置间的依赖关系
    2. 位置加权前馈网络: 通过显式地引入位置权重增强模型对位置信息的感知

    模块采用标准的Transformer结构设计，包含残差连接和层归一化。

    Args:
        d_model: 输入特征的维度
        dim_feedforward: 前馈网络的隐藏层维度
        num_heads: 多头注意力的头数
        dropout: Dropout概率
        device: 设备
    """

    def __init__(self, d_model: int, dim_feedforward: int, num_heads: int, dropout: float = 0.1, device: torch.device = None):
        super().__init__()
        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)

        # 前馈网络部分
        self.feedforward_hidden = nn.Linear(d_model, dim_feedforward)
        self.feedforward_output = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm_attention = nn.LayerNorm(d_model, device=device)
        self.norm_feedforward = nn.LayerNorm(d_model, device=device)

        # 激活函数和Dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.BoolTensor]) -> torch.Tensor:
        """
        前向传播过程，处理输入序列并增强位置信息。

        处理流程:
        1. 首先通过自注意力机制处理输入
        2. 然后通过位置加权的前馈网络
        3. 每个步骤后都进行残差连接和层归一化

        Args:
            x: 形状为(batch_size, sequence_length, d_model)的输入张量
            key_padding_mask: 用于屏蔽无效位置的布尔掩码，形状为(batch_size, sequence_length)

        Returns:
            经过位置增强后的输出张量，形状与输入相同
        """
        seq_length = x.size(1)

        # 自注意力部分
        attention_output, _ = self.attention(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        # 残差连接和归一化
        x = self.norm_attention(x + attention_output)

        # 使用序列位置作为权重增强位置信息
        position_weights = torch.arange(1, seq_length + 1, dtype=x.dtype, device=x.device).unsqueeze(1)
        feedforward_output = self.feedforward_output(self.dropout(self.activation(self.feedforward_hidden(x))) * position_weights)

        # 残差连接和归一化
        x = self.norm_feedforward(x + feedforward_output)
        return self.dropout(x)


class MidiNet(nn.Module):
    """
    Midi 音乐生成模型。

    Args:
        vocab_size: 音符词汇表大小
        d_model: 模型的特征维度
        num_heads: 多头注意力的头数
        dim_feedforward: 前馈网络的隐藏层维度
        num_layers: Transformer编码器层数
        dropout: Dropout概率
        device: 计算设备
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
        device: torch.device = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.positional_embedding = PositionalEmbedding(d_model, dim_feedforward, num_heads, dropout, device)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout=dropout, device=device, batch_first=True
        ), num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size, device=device)

        # 初始化权重
        nn.init.normal_(self.embedding.weight)
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.BoolTensor] = None):
        # 通过嵌入层
        x = self.dropout(self.embedding(x) * math.sqrt(self.vocab_size))

        # 应用位置编码
        x = self.positional_embedding(x, key_padding_mask)

        # 通过Transformer编码器
        x = self.transformer(
            x,
            mask=nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device, dtype=bool),
            src_key_padding_mask=key_padding_mask
        )

        # 输出预测
        logits = self.output_layer(x)
        return logits
