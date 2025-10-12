# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import NamedTuple, Optional
import torch
from torch import nn
from torch.nn import functional as F

AttentionKVCache = tuple[torch.Tensor, torch.Tensor]
GPTBlocksKVCache = list[AttentionKVCache]
VarianceKVCache = tuple[GPTBlocksKVCache, GPTBlocksKVCache, GPTBlocksKVCache]
NetKVCache = tuple[GPTBlocksKVCache, VarianceKVCache, Optional[GPTBlocksKVCache]]


class ScaleNorm(nn.Module):
    """
    实现了一个缩放归一化的模块。该模块根据输入的张量 `x` 计算其L2范数，并用可学习的缩放因子 `g` 对其进行缩放归一化。用于增强模型的稳定性和学习能力。

    工作流程如下：
        1. 输入张量 `x` 会计算其在最后一个维度上的L2范数。
        2. 通过一个可学习的参数 `g` 对输入进行缩放。
        3. 输出结果是经过缩放的归一化张量。

    Args:
        dim: 用于初始化缩放因子 `g` 的维度。

    Returns:
        返回缩放归一化后的张量。

    Examples:
        # 示例代码：
        scale_norm = ScaleNorm(dim=256)
        x = torch.randn(10, 256)  # 假设x的维度是[10, 256]
        output = scale_norm(x)  # 返回经过缩放归一化后的张量
    """

    def __init__(self, dim: int, device: Optional[torch.device] = None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, device=device) * (dim ** 0.5))  # 可学习的缩放因子，初始化为dim的平方根

    def forward(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp(min=torch.finfo(x.dtype).eps)  # 计算L2范数并防止为零
        return self.scale * x / norm  # 对输入张量进行缩放归一化


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

    def __init__(self, dim_head: int, num_heads: int, rope_base: float = 10000., dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dropout_rate = dropout
        dim_model = dim_head * num_heads  # 总模型维度 = 头数 * 每头维度

        # 查询、键值投影矩阵
        self.qkv_proj = nn.Linear(dim_model, dim_model * 3, device=device)

        # 输出投影矩阵，将多头输出合并回原始维度
        self.out_proj = nn.Linear(dim_model, dim_model, device=device)

        # RoPE 旋转频率
        self.inv_freq = nn.Buffer(1.0 / (rope_base ** (torch.arange(0, dim_head, 2, device=device) / dim_head)), persistent=False)

        # freqs_cis 缓存
        self.freqs_cis_cache = nn.Buffer(torch.empty(0, dim_head // 2, device=device), persistent=False)

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
        # 计算需要应用RoPE的序列长度
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
        rotated_complex = complex_tensor * freqs_cis

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


class PitchFeatureEncoderLayer(nn.Module):
    """
    音高特征编码层，基于卷积实现特征编码
    包含卷积前馈网络，使用残差连接和归一化

    工作流程：
    1. 输入张量首先通过 ScaleNorm 进行归一化
    2. 通过 ScaleNorm 归一化后，使用两个卷积层进行前馈处理
    3. 将前馈输出与原始输入进行缩放残差连接，得到最终输出

    Inputs:
        x: 输入特征张量，形状为 [batch_size, 88, dim_model]

    Outputs:
        编码后的特征张量，形状与输入相同
    """

    def __init__(self, dim_model: int, dim_feedforward: int, conv1_kernel_size: int, conv2_kernel_size: int, dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        # 前馈网络
        self.conv1 = nn.Conv1d(dim_model, dim_feedforward, conv1_kernel_size, padding="same", device=device)
        self.conv2 = nn.Conv1d(dim_feedforward, dim_model, conv2_kernel_size, padding="same", device=device)

        # 归一化与缩放
        self.feedforward_norm = ScaleNorm(dim_model, device=device)
        self.feedforward_scale = nn.Parameter(torch.zeros(1, device=device))

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        for module in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前馈网络计算
        ff_output = self.conv2(F.mish(self.conv1(self.feedforward_norm(x).transpose(1, 2)))).transpose(1, 2)
        x = x + self.dropout(ff_output * self.feedforward_scale)
        return x


class GPTBlock(nn.Module):
    """
    GPTBlock 是一个基于多头注意力和前馈网络的模块，主要用于处理序列数据。它包含自注意力机制和前馈网络，并使用缩放归一化来增强模型的稳定性。

    工作流程如下：
        1. 输入张量通过多头注意力机制进行处理，计算注意力输出。
        2. 将注意力输出与输入张量相加。
        3. 输入张量经过前馈网络处理，得到新的表示。
        4. 将前馈网络输出与注意力输出相加。

    Args:
        dim_head: 每个注意力头的维度。
        num_heads: 注意力头的数量。
        dim_feedforward: 前馈网络的隐藏层维度。
        dropout: Dropout 概率，用于防止过拟合。
        device: 可选的设备参数，用于指定模型运行的设备。

    Inputs:
        x: 输入张量，形状为 (batch_size, seq_len, dim_model)
        padding_mask: 填充掩码，形状为 (batch_size, seq_len)，用于指示哪些位置是填充的。
        kv_cache: 键值缓存，形状为 (keys_cache, values_cache)，用于增量推理。

    Outputs:
        返回处理后的张量，形状与输入相同；和键值缓存。
    """

    def __init__(self, dim_head: int, num_heads: int, dim_feedforward: int, dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        dim_model = dim_head * num_heads  # 总模型维度

        # 自注意力和前馈网络
        # PitchFeatureEncoderLayer 使用卷积层而 GPTBlock 不使用的原因是，卷积层（这里指的是因果卷积）不适合增量生成
        # 具体来说，卷积会扩充感受野，比如 conv(kernel_size=3) 需要 seq_len=3 的 x
        # 对于前馈层，两个卷积层叠加，假设两个卷积核分别是 3 和 5，第二个卷积层需要第一个卷积层的输出时间长度为 5
        # 而第二个卷积层所需要的 x[-5] 来自第一个卷积层的 x[-7]，也就是说，需要给第一个卷积层提供 seq_len=7 的 x
        # 如果是多个 GPTBlock 层叠加，计算量将会变得非常大，那么我不就增量了个寂寞？
        # 而且整个模型会变得十分 ... 逻辑混乱，让我们使用 Linear 而不是 Conv 使其保持简单
        self.attention = MultiheadAttention(dim_head, num_heads, 10000., dropout, device=device)
        self.linear1 = nn.Linear(dim_model, dim_feedforward, device=device)
        self.linear2 = nn.Linear(dim_feedforward, dim_model, device=device)

        # 归一化
        self.attention_norm = ScaleNorm(dim_model, device=device)
        self.feedforward_norm = ScaleNorm(dim_model, device=device)
        self.attention_scale = nn.Parameter(torch.zeros(1, device=device))
        self.feedforward_scale = nn.Parameter(torch.zeros(1, device=device))

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        for module in [self.linear1, self.linear2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor, kv_cache: Optional[AttentionKVCache]) -> tuple[torch.Tensor, AttentionKVCache]:
        # 多头注意力计算
        attn_output, kv_cache = self.attention(self.attention_norm(x), padding_mask, kv_cache is None, kv_cache)
        x = x + self.dropout(attn_output * self.attention_scale)

        # 前馈网络计算
        ff_output = self.linear2(F.mish(self.linear1(self.feedforward_norm(x))))
        x = x + self.dropout(ff_output * self.feedforward_scale)
        return x, kv_cache


class VariancePredictor(nn.Module):
    """
    方差预测器模块，用于预测序列数据的方差相关特征。
    本模块借鉴了 [FastSpeech 2](https://arxiv.org/abs/2006.04558) 的方差预测器思想，
    通过 GPT-2 块提取特征，最后通过线性层输出单维度的预测值。

    该模块的工作流程如下：
    1. 输入序列数据通过 GPTBlock 层进行特征提取
    2. 每个 GPTBlock 层都包含多头自注意力机制和前馈神经网络
    3. 使用残差连接和层归一化来稳定训练过程
    4. 最终通过线性输出层将特征映射为单维度预测值
    5. 支持键值缓存机制以提高推理效率

    Args:
        dim_head: 每个注意力头的维度大小
        num_heads: 注意力头的数量
        dim_feedforward: 前馈神经网络的隐藏层维度
        num_layers: GPTBlock 层的数量
        dropout: Dropout 概率，用于防止过拟合
        device: 模型运行的设备

    Inputs:
        x: 输入张量，形状为 [batch_size, seq_len, dim_model]
        padding_mask: 填充掩码，形状为 [batch_size, seq_len]
        kv_cache: 可选的键值缓存，用于加速自回归生成

    Outputs:
        x: 预测结果张量，形状为 [batch_size, seq_len]
        layers_kv_cache: 各层的键值缓存列表
    """

    def __init__(self, dim_head: int, num_heads: int, dim_feedforward: int, num_layers: int, dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        # 创建 GPT-2 块组成的序列
        self.layers = nn.ModuleList(GPTBlock(dim_head, num_heads, dim_feedforward, dropout, device) for _ in range(num_layers))

        # 输出层将多头注意力输出映射为单维度预测
        self.output_layer = nn.Linear(dim_head * num_heads, 1, device=device)

        # 初始化权重
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor, kv_cache: Optional[GPTBlocksKVCache]) -> tuple[torch.Tensor, GPTBlocksKVCache]:
        layers_kv_cache = []

        # 逐层通过 GPT-2 块
        for layer_idx, layer in enumerate(self.layers):
            x, layer_kv_cache = layer(x, padding_mask, None if kv_cache is None else kv_cache[layer_idx])
            layers_kv_cache.append(layer_kv_cache)

        # 通过输出层并压缩最后一维，得到形状 [batch_size, seq_len]
        x = self.output_layer(x).squeeze(-1)
        return x, layers_kv_cache


class MidiNetConfig(NamedTuple):
    """
    MidiNet 的配置类，包含模型的超参数设置。

    Attributes:
        pitch_dim_model: 音高特征编码器的模型主维度
        pitch_dim_feedforward: 音高特征编码器前馈网络的隐藏层维度
        num_heads: 编-解码器注意力头的数量
        dim_head: 编-解码器每个注意力头的维度
        dim_feedforward: 编-解码器前馈网络的隐藏层维度
        pitch_conv1_kernel: 音高特征编码器中第一个卷积层的卷积核大小
        pitch_conv2_kernel: 音高特征编码器中第二个卷积层的卷积核大小
        variance_bins: 音符特征离散化的精细度
        num_pitch_layers: 音高特征编码器层的数量
        num_note_count_layers: 音符数量预测器中 GPTBlock 层的数量
        num_pitch_mean_layers: 音高平均值预测器中 GPTBlock 层的数量
        num_pitch_range_layers: 音高范围预测器中 GPTBlock 层的数量
        num_encoder_layers: 编码器层的数量
        num_decoder_layers: 解码器层的数量
    """
    pitch_dim_model: int
    pitch_dim_feedforward: int
    num_heads: int
    dim_head: int
    dim_feedforward: int
    pitch_conv1_kernel: int
    pitch_conv2_kernel: int
    variance_bins: int
    num_pitch_layers: int
    num_note_count_layers: int
    num_pitch_mean_layers: int
    num_pitch_range_layers: int
    num_encoder_layers: int
    num_decoder_layers: int


class MidiNet(nn.Module):
    """
    基于 Transformer 的 MIDI 音乐生成模型，能够预测音符存在概率、音符数量、平均音高和音高范围。

    模型首先通过音高特征编码器处理钢琴卷帘输入，提取音高间的相对关系特征，然后使用编码器-解码器
    Transformer 架构进行序列建模，最后预测音符存在概率和各种音乐特征统计量。支持使用键值缓存
    加速推理过程。

    Inputs:
        x: 钢琴卷帘，[batch_size, seq_len, 88]
        note_count_target: 音符数量的目标值（用于训练）
        pitch_mean_target: 平均音高的目标值（用于训练）
        pitch_range_target: 音高范围的目标值（用于训练）
        padding_mask: 可选填充掩码，True 表示序列中的填充位置
        kv_cache: 可选 NetKVCache 对象，用于缓存键值对以加速推理
        encoder_only: 如果为 True，则仅返回编码器输出和方差预测，而不进行解码器处理

    Outputs:
        note_prediction: 预测的音符存在概率，[batch_size, seq_len, 88]
        note_count_prediction: 预测的音符数量，[batch_size, seq_len]
        pitch_mean_prediction: 预测的平均音高,[batch_size, seq_len]
        pitch_range_prediction: 预测的音高范围，[batch_size, seq_len]
        kv_cache: 编-解码器的键值对缓存

    Examples:
        >>> config = MidiNetConfig(...)
        >>> model = MidiNet(config)
        >>> input_tensor = torch.randint(0, 2, (2, 100, 88), dtype=torch.bool)
        >>> note_pred, count_pred, mean_pred, range_pred, _ = model(input_tensor, None, None, None, None, None)
        >>> print(note_pred.shape)  # torch.Size([2, 100, 88])
    """

    def __init__(self, config: MidiNetConfig, tmpname_dropout: float = 0., pitch_dropout: float = 0, encoder_dropout: float = 0., decoder_dropout: float = 0., variance_predictor_dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        dim_model = config.dim_head * config.num_heads  # 总模型维度
        self.variance_bins = config.variance_bins

        # 音符嵌入和音高聚合器
        # 这里为什么不用 nn.Embedding(88, dim_model) 呢？因为这会使模型认为 88 个音高都是不同的
        # 但事实上，我们压根没什么关注绝对音高，我们更加关注的音程信息
        # 比如 C4、G4 是纯五度关系，而 D4、A4 也是纯五度关系，如果使用 88 个不同的嵌入，模型就难以认识到它们的相对音高关系，它只会把 C4、G4、D4、A4 看成四个不同的音高
        # 所以我们需要只使用一个嵌入（区分有无音符），然后通过 RoPE 位置编码注入捕捉音程关系
        self.note_embedding = nn.Parameter(torch.Tensor(config.pitch_dim_model, device=device))
        self.pitch_feature_encoder = nn.ModuleList(PitchFeatureEncoderLayer(config.pitch_dim_model, config.pitch_dim_feedforward, config.pitch_conv1_kernel, config.pitch_conv2_kernel, pitch_dropout, device) for _ in range(config.num_pitch_layers))
        self.pitch_projection = nn.Linear(88 * config.pitch_dim_model, dim_model)

        # 编码器、解码器
        self.encoder = nn.ModuleList(GPTBlock(config.dim_head, config.num_heads, config.dim_feedforward, encoder_dropout, device) for _ in range(config.num_encoder_layers))
        self.decoder = nn.ModuleList(GPTBlock(config.dim_head, config.num_heads, config.dim_feedforward, decoder_dropout, device) for _ in range(config.num_decoder_layers))

        # 音符数量、音高平均值、音高范围预测器和嵌入
        self.note_count_predictor, self.note_count_embedding, self.pitch_mean_predictor, self.pitch_mean_embedding, self.pitch_range_predictor, self.pitch_range_embedding = [
            module
            for num_layers in [config.num_note_count_layers, config.num_pitch_mean_layers, config.num_pitch_range_layers]
            for module in [
                VariancePredictor(config.dim_head, config.num_heads, config.dim_feedforward, num_layers, variance_predictor_dropout, device=device),
                nn.Embedding(config.variance_bins, dim_model, device=device)
            ]
        ]

        # 音符预测器
        self.note_predictor = nn.Linear(dim_model, 88, device=device)

        # 初始化权重
        nn.init.uniform_(self.note_embedding)
        nn.init.zeros_(self.note_predictor.bias)
        for module in [self.note_count_embedding, self.pitch_mean_embedding, self.pitch_range_embedding, self.note_predictor]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        note_count_target: Optional[torch.Tensor] = None,
        pitch_mean_target: Optional[torch.Tensor] = None,
        pitch_range_target: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        kv_cache: Optional[NetKVCache] = None,
        encoder_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, NetKVCache]:
        # 处理输入音高张量，准备进入编码器
        batch_size, seq_len, _ = x.shape

        # 展平批次和序列维度，便于并行处理所有时间步的音高
        x = x.flatten(0, 1)  # [batch_size * seq_len, 88]

        # 将音高信息与嵌入结合，仅保留存在的音高对应的嵌入
        x = x.unsqueeze(2) * self.note_embedding.unsqueeze(0)  # [batch_size * seq_len, 88, pitch_dim_model]

        # 通过带 RoPE 的注意力机制捕获音高间的相对关系
        for layer in self.pitch_feature_encoder:
            x = layer(x)

        # 聚合音高特征
        x = self.pitch_projection(x.flatten(1, 2))  # [batch_size * seq_len, dim_model]

        # 恢复批次和序列维度
        x = x.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, dim_model]

        # 编码器
        encoder_kv_cache = []
        for layer_idx, layer in enumerate(self.encoder):
            x, layer_kv_cache = layer(x, padding_mask, None if kv_cache is None else kv_cache[0][layer_idx])
            encoder_kv_cache.append(layer_kv_cache)

        # 预测音高范围、平均值和能量
        variance_prediction, variance_kv_cache = zip(*[
            predictor(x, padding_mask, None if kv_cache is None else kv_cache[1][kv_cache_idx])
            for kv_cache_idx, predictor in enumerate([self.note_count_predictor, self.pitch_mean_predictor, self.pitch_range_predictor])
        ])

        # 如果只需要编码器输出，则直接返回
        if encoder_only:
            return torch.zeros(batch_size, seq_len, 88, device=x.device, dtype=x.dtype), *variance_prediction, (encoder_kv_cache, variance_kv_cache, None)

        # 使用目标值替代预测值（如果提供）
        variance = [
            prediction if target is None else target
            for prediction, target in zip(variance_prediction, [note_count_target, pitch_mean_target, pitch_range_target])
        ]

        # 限制范围并离散化
        note_count, pitch_mean, pitch_range = [
            (item.clamp(min=0, max=1) * (self.variance_bins - 1)).round().to(dtype=int)
            for item in variance
        ]

        # 将音高和能量作为附加特征添加到解码器输入中
        x = x + self.note_count_embedding(note_count) + self.pitch_mean_embedding(pitch_mean) + self.pitch_range_embedding(pitch_range)

        # 解码器
        decoder_kv_cache = []
        for layer_idx, layer in enumerate(self.decoder):
            x, layer_kv_cache = layer(x, padding_mask, None if kv_cache is None else kv_cache[2][layer_idx])
            decoder_kv_cache.append(layer_kv_cache)

        # 激活音符预测
        note_prediction = self.note_predictor(x)  # [batch_size, seq_len, 88]
        return note_prediction, *variance_prediction, (encoder_kv_cache, variance_kv_cache, decoder_kv_cache)
