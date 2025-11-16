# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import NamedTuple, Optional
import torch
from torch import nn
from torch.nn import functional as F

AttentionKVCache = tuple[torch.Tensor, torch.Tensor]
NetKVCache = list[AttentionKVCache]


class MultiheadAttention(nn.Module):
    """
    多头注意力机制模块，支持旋转位置编码(RoPE)和键值缓存

    该模块实现了基于缩放点积的多头注意力机制，包含以下功能：
    1. 将输入投影到查询、键、值空间
    2. 应用旋转位置编码(RoPE)增强位置感知能力
    3. 支持键值缓存以提高增量推理效率
    4. 处理填充掩码和因果掩码
    5. 使用缩放点积注意力计算注意力权重
    6. 将多头输出合并回原始维度

    Inputs:
        qkv: 查询、键、值张量，形状为 [batch_size, seq_len, dim_model]
        padding_mask: 填充掩码，形状为 [batch_size, seq_len]，True 表示填充位置
        is_causal: 是否使用因果掩码
        kv_cache: 键值缓存元组 (keys_cache, values_cache)，仅应在推理阶段使用

    Outputs:
        output: 注意力输出张量，形状为 [batch_size, seq_len_q, dim_model]
        updated_kv_cache: 更新后的键值缓存元组
    """

    def __init__(self, dim_head: int, num_heads: int, rope_base: float = 10000, dropout: float = 0):
        super().__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dropout_rate = dropout
        dim_model = dim_head * num_heads  # 总模型维度 = 头数 * 每头维度

        # 查询、键值投影矩阵
        self.qkv_proj = nn.Linear(dim_model, dim_model * 3)

        # 输出投影矩阵，将多头输出合并回原始维度
        self.out_proj = nn.Linear(dim_model, dim_model)

        # RoPE 旋转频率
        self.inv_freq = nn.Buffer(1.0 / (rope_base ** (torch.arange(0, dim_head, 2) / dim_head)), persistent=False)

        # freqs_cis 缓存
        self.freqs_cis_cache = nn.Buffer(torch.empty(0, dim_head // 2), persistent=False)

        # 使用 Xavier 均匀分布初始化查询、键、值、投影权重
        for module in [self.qkv_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def apply_rope(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        """
        应用旋转位置编码(RoPE)到输入张量。

        旋转位置编码通过复数乘法实现位置信息的注入：
        1. 将输入张量的最后两个维度视为复数对(实部和虚部)
        2. 生成与位置相关的旋转复数向量
        3. 通过复数乘法实现旋转操作
        4. 将旋转后的复数转换回实数表示

        Args:
            x: 输入张量，形状为 [batch, num_heads, seq_len, dim_head]
            offset: 位置偏移量，用于增量推理时调整位置索引

        Returns:
            应用旋转位置编码后的张量，形状与输入相同
        """
        # 计算需要应用 RoPE 的序列长度
        required_seq_len = x.size(2) + offset

        # 检查并更新旋转频率缓存
        current_cache_len = self.freqs_cis_cache.size(0)
        if current_cache_len < required_seq_len:
            # 生成缺失位置的时间索引
            new_positions = torch.arange(
                current_cache_len,
                required_seq_len,
                device=self.inv_freq.device
            )
            # 计算新位置的旋转频率
            new_freqs = torch.outer(new_positions, self.inv_freq)
            # 转换为复数形式 (cosθ + i·sinθ)
            new_cis = torch.polar(torch.ones_like(new_freqs), new_freqs)
            # 更新缓存
            self.freqs_cis_cache = torch.cat([self.freqs_cis_cache, new_cis], dim=0)

        # 获取当前序列所需的旋转频率
        freqs_cis = self.freqs_cis_cache[offset:required_seq_len]

        # 将最后维度重塑为复数对 (..., dim_head//2, 2)
        # 这里转换为 float 是因为半精度复数运算不被支持
        complex_shape = x.shape[:-1] + (-1, 2)
        complex_pairs = x.float().reshape(complex_shape)

        # 转换为复数张量
        complex_tensor = torch.view_as_complex(complex_pairs)

        # 调整旋转频率形状以匹配输入 (添加批量和头维度)
        freqs_cis = freqs_cis.view(1, 1, -1, freqs_cis.shape[-1])

        # 应用旋转（复数乘法）
        rotated_complex = complex_tensor * freqs_cis.clone()

        # 转换回实数表示
        rotated_real = torch.view_as_real(rotated_complex)

        # 展平最后两个维度 (..., dim_head//2, 2) -> (..., dim_head)
        rotated_output = rotated_real.flatten(-2).to(dtype=x.dtype)
        return rotated_output

    def forward(
        self,
        qkv: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor],
        is_causal: bool,
        kv_cache: Optional[AttentionKVCache]
    ) -> tuple[torch.Tensor, AttentionKVCache]:
        batch_size, seq_len, dim_model = qkv.shape

        # 计算查询、键值投影 [batch_size, seq_len, dim_model]
        queries, keys, values = self.qkv_proj(qkv).chunk(3, dim=-1)  # 分割为 Q, K, V

        # 调整查询、键、值形状为 [batch_size, seq_len, num_heads, dim_head]，
        # 并重排维度为 PyTorch 注意力要求的形状 [batch_size, num_heads, seq_len, dim_head]
        queries, keys, values = [x.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2) for x in [queries, keys, values]]

        # 应用 RoPE 到 Q/K
        cache_len = 0 if kv_cache is None else kv_cache[0].size(2)
        queries = self.apply_rope(queries, cache_len)
        keys = self.apply_rope(keys, cache_len)

        # 拼接键值缓存，形状为 [batch_size, num_heads, seq_len_kv, dim_head]
        if kv_cache is not None:
            keys = torch.cat([kv_cache[0], keys], dim=2)
            values = torch.cat([kv_cache[1], values], dim=2)

        # 计算填充掩码和因果掩码，形状为 [batch, seq_len_q, seq_len_kv]
        attn_mask = torch.zeros(batch_size, seq_len, keys.size(2), dtype=bool, device=qkv.device)
        if padding_mask is not None:
            attn_mask |= padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
        if is_causal:
            attn_mask |= ~torch.tril(torch.ones(seq_len, seq_len, dtype=bool, device=qkv.device))

        # 显式转化为 float 类型掩码，避免版本兼容性问题
        # 比如在大部分包（transformers 等），True 表示填充
        # 但在 torch.nn.functional.scaled_dot_product_attention 中，True 表示允许注意力
        attn_mask = torch.where(attn_mask, -torch.inf, 0.0)

        # 扩展掩码以匹配多头注意力的要求，形状为 [batch, 1, seq_len_q, seq_len_kv]
        attn_mask = attn_mask.unsqueeze(1)

        # 计算缩放点积注意力
        attn_output = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_mask,
            dropout_p=(self.dropout_rate if self.training else 0.),
        )

        # 合并多头输出 (batch, seq_len_q, dim_model)
        return self.out_proj(attn_output.transpose(1, 2).reshape(batch_size, -1, dim_model)), (keys, values)


class GPTBlock(nn.Module):
    """
    GPT 模型的基础构建块，包含多头自注意力机制和前馈神经网络
    采用预归一化架构，包含残差连接和可学习的缩放因子

    工作流程：
    1. 对输入进行层归一化后计算多头自注意力
    2. 将注意力输出与输入残差连接，应用 dropout 和可学习缩放
    3. 对结果进行层归一化后通过门控前馈网络
    4. 将前馈网络输出与残差连接，应用可学习缩放

    Inputs:
        x: 输入张量，形状为 [batch_size, seq_len, hidden_dim]
        padding_mask: 可选的填充掩码，形状为 [batch_size, seq_len]
        kv_cache: 可选的键值缓存，用于推理时的增量计算

    Outputs:
        变换后的张量和键值缓存
    """

    def __init__(self, dim_head: int, num_heads: int, dim_feedforward: int, dropout: float = 0.):
        super().__init__()
        dim_model = dim_head * num_heads  # 总模型维度

        # 自注意力和前馈网络
        self.attention = MultiheadAttention(dim_head, num_heads, 10000., dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward * 2)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        # 归一化与缩放
        self.attention_norm = nn.LayerNorm(dim_model)
        self.feedforward_norm = nn.LayerNorm(dim_model)
        self.attention_scale = nn.Parameter(torch.ones(1) * 1e-5)
        self.feedforward_scale = nn.Parameter(torch.ones(1) * 1e-5)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        for module in [self.linear1, self.linear2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.BoolTensor], kv_cache: Optional[AttentionKVCache]) -> tuple[torch.Tensor, AttentionKVCache]:
        # 多头注意力计算
        attn_output, kv_cache = self.attention(self.attention_norm(x), padding_mask, kv_cache is None, kv_cache)
        x = x + attn_output * self.attention_scale

        # 前馈网络计算
        gate, value = self.linear1(self.feedforward_norm(x)).chunk(2, dim=-1)  # 获取并分离 gate 和 value
        x = x + self.linear2(self.dropout(value * F.silu(gate))) * self.feedforward_scale  # 残差连接
        return x, kv_cache


class GPTConfig(NamedTuple):
    """
    GPT 的配置类，包含模型的超参数设置。

    Attributes:
        vocab_size: 词汇表大小
        num_heads: 注意力头的数量
        dim_head: 每个注意力头的维度
        dim_feedforward: 前馈网络的隐藏层维度
        num_layers: GPTBlock 堆叠层数
    """
    vocab_size: int
    num_heads: int
    dim_head: int
    dim_feedforward: int
    num_layers: int


class GPT(nn.Module):
    """
    仿照 GPT 构造的模块

    Inputs:
        x: 序列，形状为 [batch_size, seq_len]
        padding_mask: 填充掩码，形状为 [batch_size, seq_len]，True 表示填充位置
        kv_cache: 键值缓存元组 (keys_cache, values_cache)，仅应在推理阶段使用

    Outputs:
        output: 序列对应位置的预测，形状为 [batch_size, seq_len_q, vocab_size]
        updated_kv_cache: 更新后的键值缓存元组
    """

    def __init__(self, config: GPTConfig, dropout: float = 0.):
        super().__init__()
        self.dim_model = config.dim_head * config.num_heads  # 总模型维度

        # 相对音高嵌入
        self.embedding = nn.Embedding(config.vocab_size, self.dim_model)

        # GPTBlock 堆叠
        self.layers = nn.ModuleList(GPTBlock(config.dim_head, config.num_heads, config.dim_feedforward, dropout) for _ in range(config.num_layers))

        # 音符预测器
        self.predictor = nn.Linear(self.dim_model, config.vocab_size)

        # 初始化权重
        nn.init.zeros_(self.predictor.bias)
        for module in [self.embedding, self.predictor]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x: torch.LongTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        kv_cache: Optional[NetKVCache] = None,
    ) -> tuple[torch.Tensor, NetKVCache]:
        # 经过嵌入层
        x = self.embedding(x)  # [batch_size, seq_len, dim_model]
        x = x * x.size(-1) ** 0.5  # 缩放嵌入

        # 经过 GPTBlock 堆叠
        layers_kv_cache = []
        for layer_idx, layer in enumerate(self.layers):
            layer_kv_cache = None if kv_cache is None else kv_cache[layer_idx]
            x, layer_kv_cache = layer(x, padding_mask, layer_kv_cache)
            layers_kv_cache.append(layer_kv_cache)

        # 经过音符预测器
        x = self.predictor(x)  # [batch_size, seq_len, vocab_size]
        return x, layers_kv_cache
