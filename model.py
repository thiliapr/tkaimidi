# 模型的代码
# Copyright (C)  thiliapr 2024-2025
# License: AGPLv3-or-later

import math
import json
import base64
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

TIME_PRECISION = 120  # 时间精度，表示每个音符的最小时间单位
MAX_TIME_DIFF = 960  # 两个音符之间允许的最大时间差
NOTE_DURATION_COUNT = MAX_TIME_DIFF // TIME_PRECISION + 1  # 允许的音符时间长度的数量
DEFAULT_LENGTH = 4096


class PositionalEncoding(nn.Module):
    r"""
    位置编码

    来源: https://github.com/pytorch/examples/blob/c0b889d5f43150f288ecdd5dbd16c146d79e5bdf/word_language_model/model.py#L65
    注入一些关于序列中标记的相对或绝对位置的信息。
    位置编码与嵌入的维度相同，以便可以将两者相加。
    在这里，我们使用不同频率的正弦和余弦函数。

    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^{(2i/d_model)})
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^{(2i/d_model)})
        \text{其中 pos 是单词位置，i 是嵌入索引}
    Args:
        d_model: 嵌入维度。
        dropout: dropout 值。
        max_len: 输入序列的最大长度。
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, length=DEFAULT_LENGTH):
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.register_buffer("pe", self._generate_pe(length), persistent=False)

    def _generate_pe(self, length: int):
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 保持为 (1, length, d_model)
        return pe

    def forward(self, x):
        # x 的形状为 (batch_size, seq_len, d_model)
        seq_len = x.size(1)  # 获取序列长度
        if seq_len < self.pe.size(1):
            self.pe = self._generate_pe(seq_len)
        x = x + self.pe[:, :seq_len, :]  # 进行位置编码
        return F.dropout(x, self.dropout)


class MidiBlock(nn.Module):
    """
    类似`TransformerDecoderLayer`的模块，但没有编码器-解码器注意力层。
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = dropout
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, tgt_mask):
        x = self.attention(x, x, x, attn_mask=tgt_mask, need_weights=False)[0] + x
        x = self.norm1(x)
        x = F.dropout(self.linear2(F.relu(self.linear1(x))), self.dropout) + x
        x = self.norm2(x)
        return x


class MidiNet(nn.Module):
    """
    基于音符和时间信息，预测下一个音符的时间和音符编号的神经网络模型。
    """

    def __init__(self):
        super().__init__()
        vocab_size = 128 * NOTE_DURATION_COUNT
        self.d_model = 768

        self.embedding = nn.utils.skip_init(nn.Embedding, vocab_size, self.d_model)  # 嵌入层
        self.pos_encoder = PositionalEncoding(self.d_model, 0.1)
        self.blocks = nn.ModuleList(MidiBlock(self.d_model, 8, 2048, 0.1) for _ in range(6))  # Transformer 解码器层堆叠
        self.fc_out = nn.utils.skip_init(nn.Linear, self.d_model, vocab_size)  # 将嵌入映射到词汇大小

        self.register_buffer("last_mask", torch.tensor([]), persistent=False)  # CasualMask 初始化
        self.init_weights()  # 初始化权重

    def init_weights(self):
        nn.init.normal_(self.embedding.weight)
        nn.init.kaiming_normal_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x, mask: bool = True):
        x = self.embedding(x) * math.sqrt(self.d_model)  # 输入通过嵌入层
        x = self.pos_encoder(x)  # 输入通过位置编码
        if mask:
            if x.size(1) != self.last_mask.size(0):
                self.last_mask = torch.tril(torch.ones(x.size(1), x.size(1)), diagonal=1) == 0  # 生成CasualMask
            self.last_mask = self.last_mask.to(x.device)
            tgt_mask = self.last_mask
        else:
            tgt_mask = None

        for block in self.blocks:
            x = block(x, tgt_mask=tgt_mask)
        logits = self.fc_out(x)  # 预测下一个 token
        return logits


def save_checkpoint(model: MidiNet, optimizer: optim.SGD, train_loss: list[float], val_loss: list[float], train_accuracy: list[float], val_accuracy: list[float], dataset_length: int, train_start: int, last_batch: int, generator_state: torch.tensor, path: pathlib.Path):
    """
    保存模型的检查点到指定路径，包括模型的权重以及训练的进度信息。

    Args:
        model: 要保存的模型实例
        optimizer: 要保存的优化器实例
        train_loss: 训练损失
        val_loss: 验证损失
        train_accuracy: 训练集准确率
        val_accuracy: 验证集准确率
        dataset_length: 数据集的长度
        train_start: 拆分数据集时训练集的开始索引
        last_batch: 上次训练时的未训练完成的epoch训练了多少个batch
        generator_state: 训练数据的索引随机采样器的生成器的状态
        path: 保存检查点的目录路径
    """
    path.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在，如果不存在则创建
    model = model.cpu()  # 将模型移到CPU进行保存
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()  # 处理DataParallel情况
    torch.save(model_state_dict, path / "model.pth")  # 保存模型权重
    torch.save(optimizer.state_dict(), path / "optimizer.pth")  # 保存优化器权重
    # 保存训练信息
    with open(path / "train_info.json", "w") as f:
        json.dump(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "dataset_length": dataset_length,
                "train_start": train_start,
                "last_batch": last_batch,
                "generator_state": base64.b64encode(bytes(generator_state.tolist())).decode(),
            },
            f,
        )  # 将训练信息写入JSON文件


def load_checkpoint(path: pathlib.Path, train: bool = False):
    """
    从指定路径加载模型的检查点，并恢复训练状态。

    Args:
        path: 加载检查点的目录路径
        train: 是否加载训练所需数据（优化器状态等）

    Returns:
        train关闭时: 模型的状态
        train启用时: 模型和优化器的状态，训练、验证的损失和准确率的历史记录，上一次数据集的长度，拆分数据集时训练集的开始索引，训练数据的索引随机采样器的生成器的状态，上次训练时的未训练完成的epoch训练了多少个batch
    """
    # 检查并加载模型权重
    model_state = {}
    if (model_path := path / "model.pth").exists():
        # 加载模型的状态字典并更新
        model_state = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))  # 从检查点加载权重

    if train:
        # 检查并加载优化器权重
        optimizer_state = {}
        if (optimizer_path := path / "optimizer.pth").exists():
            optimizer_state = torch.load(optimizer_path, weights_only=True, map_location=torch.device("cpu"))  # 从检查点加载权重

        # 尝试加载训练信息文件
        train_info_path = path / "train_info.json"
        if train_info_path.exists():
            with open(train_info_path, "r") as f:
                train_info = json.load(f)  # 读取训练信息
        else:
            train_info = {}  # 如果文件不存在，返回空字典

        if "generator_state" in train_info:
            generator_state = torch.tensor(list(base64.b64decode(train_info["generator_state"])), dtype=torch.uint8)
        else:
            generator_state = ...

        # 返回训练损失和验证损失
        return model_state, optimizer_state, train_info.get("train_loss", []), train_info.get("val_loss", []), train_info.get("train_accuracy", []), train_info.get("val_accuracy", []), train_info.get("dataset_length", -1), train_info.get("train_start", 0), train_info.get("last_batch", 0), generator_state

    return model_state
