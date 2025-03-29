# 模型的定义、模型与恢复训练所需信息的加载和保存。
"""
模型的定义、模型与恢复训练所需信息的加载和保存。

本模块定义了一个基于Transformer架构的神经网络模型，用于音乐生成任务。该模块还提供了模型的保存和加载功能，以便在训练过程中保存模型状态和训练信息。

使用示例:
- 创建模型实例: `model = MidiNet()`
- 保存模型检查点: `save_checkpoint(model, optimizer, train_loss, val_loss, train_accuracy, val_accuracy, dataset_length, train_start, path)`
- 加载模型检查点:
  - `model_state, optimizer_state, train_loss, val_loss, train_accuracy, val_accuracy, dataset_length, train_start = load_checkpoint(path, train=True)`
  - `model = load_checkpoint(path, train=False)`
"""

# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import math
import json
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim

TIME_PRECISION = 120  # 时间精度，表示每个音符的最小时间单位
VOCAB_SIZE = 12 + 12 + 12 + 3 + 3 + 2


class MusicEventEmbedding(nn.Module):
    r"""
    音乐事件嵌入层，包含词嵌入和位置编码

    来源: https://github.com/pytorch/examples/blob/c0b889d5f43150f288ecdd5dbd16c146d79e5bdf/word_language_model/model.py#L65
    注入一些关于序列中标记的相对或绝对位置的信息。
    位置编码与嵌入的维度相同，以便可以将两者相加。
    在这里，我们使用不同频率的正弦和余弦函数。

    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^{(2i/d_model)})
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^{(2i/d_model)})
        \text{其中 pos 是单词位置，i 是维度索引}
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1, device=torch.device("cpu")):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, device=device)

        # 初始化位置编码缓冲区
        self.register_buffer("positional_encoding", torch.tensor([]), persistent=False)

        # 初始化权重
        nn.init.normal_(self.token_embedding.weight)

    def _generate_positional_encoding(self, max_length: int, device=torch.device("cpu")):
        "生成位置编码矩阵"
        position = torch.arange(0, max_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=device).float() * (-math.log(10000.0) / self.embedding_dim))
        pe = torch.zeros(max_length, self.embedding_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用cos
        return pe

    def _detect_event_changes(self, input_sequence: torch.Tensor):
        """
        检测输入序列中的事件变化点
        返回每个batch中事件变化的索引列表
        """
        batch_event_changes = []

        for sequence in input_sequence:
            event_changes = []
            note_counter = 0

            for i in range(1, len(sequence)):
                current_event = sequence[i]
                previous_event = sequence[i - 1]

                # 处理特殊控制事件
                if 36 <= current_event <= 41:
                    note_counter = 2

                # 检测事件类型变化
                if current_event < 12:  # 音符事件
                    if previous_event >= 12 or note_counter > 0:  # 前一个是非音符或处于控制事件中
                        event_changes.append(i)
                    note_counter -= 1
                elif previous_event < 12:  # 非音符事件且前一个是音符
                    event_changes.append(i)

            batch_event_changes.append(event_changes)

        return batch_event_changes

    def forward(self, input_tokens: torch.Tensor):
        """
        前向传播
        1. 检测事件变化点
        2. 应用词嵌入
        3. 应用位置编码
        4. 应用Dropout
        """
        # 检测事件变化点
        event_change_indices = self._detect_event_changes(input_tokens)

        # 确保位置编码足够长
        max_events_needed = max(len(changes) for changes in event_change_indices) + 1
        if max_events_needed > self.positional_encoding.size(0):
            self.positional_encoding = self._generate_positional_encoding(max_events_needed, input_tokens.device)

        # 应用词嵌入
        embeddings = self.token_embedding(input_tokens) * math.sqrt(self.embedding_dim)

        # 应用位置编码
        for batch_idx, changes in enumerate(event_change_indices):
            if not changes:
                embeddings[batch_idx] += self.positional_encoding[0]
                continue

            # 第一个事件段
            embeddings[batch_idx, :changes[0]] += self.positional_encoding[0]

            # 中间事件段
            for i, pos in enumerate(changes):
                next_pos = changes[i + 1] if i + 1 < len(changes) else None
                if next_pos:
                    embeddings[batch_idx, pos:next_pos] += self.positional_encoding[i + 1]
                else:
                    embeddings[batch_idx, pos:] += self.positional_encoding[i + 1]

        # 应用Dropout
        return self.dropout(embeddings)


class MidiNet(nn.Module):
    """
    基于音符和时间信息，预测下一个音符的时间和音符编号的神经网络模型。
    """

    def __init__(self, dropout: float = 0.1, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.embedding_dim = 768
        self.num_heads = 12
        self.ff_dim = 1536
        self.num_layers = 12

        # 嵌入层
        self.event_embedding = MusicEventEmbedding(VOCAB_SIZE, self.embedding_dim, dropout, device=device)

        # Transformer编码器层
        self.transformer_layers = nn.ModuleList(nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=dropout,
            batch_first=True,
            device=device
        ) for _ in range(self.num_layers))

        # 输出层
        self.output_layer = nn.Linear(self.embedding_dim, VOCAB_SIZE, device=device)

        # 缓存因果掩码
        self.register_buffer("causal_mask_cache", torch.tensor([], device=device), persistent=False)

        # 初始化权重
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def _get_causal_mask(self, sequence_length: int, device: torch.device):
        "获取因果掩码"
        if self.causal_mask_cache.size(0) != sequence_length:
            self.causal_mask_cache = torch.triu(torch.ones(sequence_length, sequence_length, device=device), diagonal=1).bool()
        return self.causal_mask_cache

    def forward(self, input_tokens: torch.Tensor, use_causal_mask: bool = True):
        """
        前向传播

        Args:
            input_tokens: 输入token序列 (batch_size, seq_len)
            use_causal_mask: 是否使用因果掩码
        """
        # 准备掩码
        causal_mask = self._get_causal_mask(input_tokens.size(1), input_tokens.device) if use_causal_mask else None

        # 通过嵌入层
        x = self.event_embedding(input_tokens)

        # 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x, src_mask=causal_mask)

        # 输出预测
        logits = self.output_layer(x)
        return logits


def save_checkpoint(model: MidiNet, optimizer: optim.AdamW, train_loss: list[list[float]], val_loss: list[float], train_accuracy: list[float], val_accuracy: list[float], dataset_length: int, train_start: int, path: pathlib.Path):
    """
    保存模型的检查点到指定路径，包括模型的权重以及训练的进度信息。

    Args:
        model: 要保存的模型实例
        optimizer: 要保存的优化器实例
        train_loss: 每一个Epoch的每一步训练损失
        val_loss: 验证损失
        train_accuracy: 训练集准确率
        val_accuracy: 验证集准确率
        dataset_length: 数据集的长度
        train_start: 拆分数据集时训练集的开始索引
        path: 保存检查点的目录路径
    """
    path.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在，如果不存在则创建

    model = model.cpu()  # 将模型移到CPU进行保存
    # 处理DataParallel情况
    if isinstance(model, nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), path / "model.pth")  # 保存模型权重
    torch.save(optimizer.state_dict(), path / "optimizer.pth")  # 保存优化器权重

    # 保存训练信息
    with open(path / "train_info.json", "w") as f:
        json.dump(
            {
                "dataset_length": dataset_length,
                "train_start": train_start,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "train_loss": train_loss,
                "val_loss": val_loss,
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
        train启用时: 模型和优化器的状态，训练、验证的损失和准确率的历史记录，上一次数据集的长度，拆分数据集时训练集的开始索引
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

        # 返回训练损失和验证损失
        return model_state, optimizer_state, train_info.get("train_loss", []), train_info.get("val_loss", []), train_info.get("train_accuracy", []), train_info.get("val_accuracy", []), train_info.get("dataset_length", -1), train_info.get("train_start", 0)

    return model_state
