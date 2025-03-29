# 模型的定义、模型与恢复训练所需信息的加载和保存。
"""
模型的定义、模型与恢复训练所需信息的加载和保存。

本模块定义了一个基于LSTM架构的神经网络模型，用于音乐生成任务。该模块还提供了模型的保存和加载功能，以便在训练过程中保存模型状态和训练信息。

使用示例:
- 创建模型实例: `model = MidiNet()`
- 保存模型检查点: `save_checkpoint(model, optimizer, train_loss, val_loss, train_accuracy, val_accuracy, dataset_length, train_start, path)`
- 加载模型检查点:
  - `model_state, optimizer_state, train_loss, val_loss, train_accuracy, val_accuracy, dataset_length, train_start = load_checkpoint(path, train=True)`
  - `model_state = load_checkpoint(path, train=False)`
"""

# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import json
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim

TIME_PRECISION = 120  # 时间精度，表示每个音符的最小时间单位
VOCAB_SIZE = 12 + 12 + 12 + 3 + 3 + 2  # 词汇库大小
EMBEDDING_DIM = 768
HIDDEN_SIZE = 1536
NUM_LAYERS = 3


class MidiNet(nn.Module):
    """
    基于音符和时间信息，预测下一个音符的时间和音符编号的神经网络模型。
    """

    def __init__(self, dropout: float = 0.1, device: torch.device = torch.device("cpu")):
        super().__init__()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 嵌入层
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, device=device)

        # LSTM
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True, dropout=dropout, device=device)

        # 输出层
        self.output_layer = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, device=device)

        # 初始化权重
        nn.init.normal_(self.embedding.weight)
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, input_tokens: torch.Tensor):
        """
        前向传播

        Args:
            input_tokens: 输入token序列 (batch_size, seq_len)
        """
        # 通过嵌入层
        x = self.dropout(self.embedding(input_tokens))

        # 通过LSTM层
        x = self.dropout(self.lstm(x)[0])

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
