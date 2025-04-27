"MIDI 音乐生成模型训练模块"

# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import pathlib
import random
import mido
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import DataParallel
from transformers import PreTrainedTokenizerFast
from typing import Iterator

import warnings
import os
from multiprocessing import cpu_count

# 忽略警告
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*deprecated.*")
# 解除线程数量限制
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count())
torch.set_num_threads(cpu_count())

# 根据是否在 Jupyter 环境下导入不同库
if "get_ipython" in globals():
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
    from model import MidiNet
    from checkpoint import load_checkpoint, save_checkpoint
    from utils import midi_to_notes, notes_to_sheet, empty_cache
    from tokenizer import data_to_str


class MidiDataset(Dataset):
    """
    MIDI 数据集类，用于加载和处理 MIDI 文件，将其转化为模型可以使用的格式。

    该类的功能包括:
    1. 读取指定目录下的所有 MIDI 文件。
    2. 将每个 MIDI 文件转化为音符序列。
    3. 将音符序列分割为指定大小的子序列。
    4. 提供数据集大小和索引功能，方便模型训练使用。

    Attributes:
        data: 存储所有 MIDI 文件的音符数据，每个元素为一个 MIDI 文件的音符序列。
        sequences: 存储每个 MIDI 文件中，音符序列的开始位置和音符数量，用于切分子序列。
        tokenizer: 用于将音符序列转换为模型输入的分词器。

    Args:
        midi_dirs: 存放 MIDI 文件的目录列表。
        tokenizer: 用于音符序列转化为模型输入的分词器。
        min_sequence_length: 每个子序列的最小长度。
        show_progress: 是否显示加载进度条。

    Examples:
        >>> midi_dirs = [pathlib.Path("/path/to/midi/files")]
        >>> tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_name")
        >>> dataset = MidiDataset(midi_dirs=midi_dirs, tokenizer=tokenizer)
        >>> len(dataset)  # 返回数据集的子序列数量
        100
    """

    def __init__(self, midi_dirs: list[pathlib.Path], tokenizer: PreTrainedTokenizerFast, min_sequence_length: int, show_progress: bool = True):
        self.data = []  # 存储每个 MIDI 文件的数据
        self.sequences = []  # 存储每个序列的文件索引、偏移量及其音符数量
        self.tokenizer = tokenizer

        # 获取所有 MIDI 文件
        midi_files = [file for midi_dir in midi_dirs for file in midi_dir.glob("**/*.mid")]

        # 如果需要显示进度条
        if show_progress:
            progress_bar = tqdm(desc="加载音乐数据集", total=len(midi_files))

        for i, filepath in enumerate(midi_files):
            # 读取并转化 MIDI 文件
            notes = midi_to_notes(mido.MidiFile(filepath, clip=True))
            sheet, positions = notes_to_sheet(notes)

            # 将每个音符序列切分为子序列
            self.sequences.extend(
                # 保存每个子序列的相关信息: 当前 MIDI 文件的索引、起始位置，以及子序列的长度
                ((i, positions[offset]), len(notes) - offset) for offset in range(max(1, len(notes) - min_sequence_length))
                if offset == 0 or notes[offset][1]  # 音符的起始点或音符是与前一个音符有时间间隔
            )

            # 将当前 MIDI 文件的音符数据加入到 data 列表中
            self.data.append(data_to_str(sheet))

            # 更新进度条
            if show_progress:
                progress_bar.update()

        # 关闭进度条
        if show_progress:
            progress_bar.close()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index: int):
        (file_index, offset), _ = self.sequences[index]
        sequence = self.tokenizer.encode(self.data[file_index][offset:])
        return torch.tensor(sequence[:-1], dtype=torch.long), torch.tensor(sequence[1:], dtype=torch.long)


class MidiDatasetSampler(Sampler[int]):
    """
    MIDI 数据集采样器类，用于从 MIDI 数据集中按批次生成索引。

    该类的功能包括:
    1. 根据 MIDI 数据集的音符序列长度生成批次。
    2. 每个批次的样本数量和总长度会满足指定的 `max_batch_size` 参数。
    3. 提供批次数据的迭代功能，便于模型训练时使用。

    Args:
        dataset: 一个包含 MIDI 数据的 `MidiDataset` 实例。
        max_batch_size: 每个批次的序列长度的平方和上限。
        drop_last: 如果最后一个批次的序列长度的平方小于 `max_batch_size`，则丢弃该批次。

    Returns:
        一个生成器，每次迭代返回一个包含批次索引的列表。

    Examples:
        >>> dataset = MidiDataset(midi_dirs=["/path/to/midi/files"], tokenizer=tokenizer)
        >>> sampler = MidiDatasetSampler(dataset=dataset, max_batch_size=2 * 768 ** 2)
        >>> for batch in sampler:
        >>>     print(batch)  # 打印每个批次的索引
    """

    def __init__(self, dataset: MidiDataset, max_batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.drop_last = drop_last

    def __iter__(self):
        # 获取数据集的长度
        length = len(self.dataset)

        # 初始化批次列表
        batches = []
        batch = []
        cur_batch_size = 0

        # 先按序列长度排序，如果序列长度相同就随机排序
        for index, sequence_length in sorted(
            [(index, sequence_length) for index, (_, sequence_length) in enumerate(self.dataset.sequences)],
            key=lambda x: x[1] * length + random.randint(0, length - 1)
        ):
            # 增加样本
            batch.append(index)
            cur_batch_size += sequence_length ** 2

            # 如果当前批次的长度平方和大于 max_batch_size，则保存并清空当前批次
            if cur_batch_size > self.max_batch_size:
                batches.append(batch.copy())
                cur_batch_size = 0
                batch.clear()

        # 如果不丢弃不满足条件的批次，则将其加入批次列表
        if batch and not self.drop_last:
            batches.append(batch)

        # 打乱所有批次顺序
        random.shuffle(batches)

        # 返回每个批次
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.dataset)


def sequence_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_token: int):
    "将多个样本合成统一长度的batch。"
    inputs, labels = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token)
    return inputs, labels


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor, pad_token: int) -> float:
    "计算准确率。"
    mask = labels != pad_token
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


def train(
    model: MidiNet,
    dataset: MidiDataset,
    optimizer: optim.Adam,
    vocab_size: int,
    num_epochs: int,
    max_batch_size: int = 1,
    pad_token: int = 0,
    device: torch.device = None,
) -> Iterator[tuple[list[int], int]]:
    """
    训练模型的函数。

    此函数进行多轮训练，逐步优化模型参数，输出每个epoch的训练损失和准确率。

    工作流程:
        1. 初始化数据加载器。
        2. 将模型移动到指定设备。
        3. 选择交叉熵损失函数。
        4. 使用进度条显示训练进度。
        5. 在每个epoch中:
            a. 切换模型到训练模式。
            b. 对每个batch进行前向传播、计算损失、反向传播和更新参数。
            c. 累积损失和准确率。
            d. 返回这个epoch每一步的训练损失和准确率平均值。

    Args:
        model: 需要训练的神经网络模型。
        dataset: 包含训练数据的Dataset对象。
        optimizer: 用于优化模型的优化器。
        vocab_size: 词汇表的大小，用于调整输出层的维度。
        num_epochs: 训练的轮数。
        max_batch_size: 每个批次的序列长度的平方和上限。
        pad_token: 填充token的标记，用于忽略计算损失。
        device: 指定训练的设备。

    Yields:
        该epoch每一步的训练损失和训练准确率的平均值。

    Examples:
        >>> list(train(model, dataset, optimizer))
        [([1.9, 0.89, 0.6, 0.4], 0.8), ...]
    """
    empty_cache()  # 清理缓存以释放内存

    # 获取采样器，按 batch 划分数据
    sampler = MidiDatasetSampler(dataset, max_batch_size)

    # 获取训练数据加载器
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda x: sequence_collate_fn(x, pad_token=pad_token))

    # 将模型移动到设备
    model = model.to(device)

    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    # 创建进度条，显示训练进度
    progress_bar = tqdm(total=len(dataloader) * num_epochs)

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        train_loss, train_acc = [], []  # 初始化损失、准确率列表
        progress_bar.set_description(f"训练第 {epoch + 1} 个 Epoch")

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs, key_padding_mask=inputs == pad_token).view(-1, vocab_size)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            train_acc.append(compute_accuracy(torch.argmax(outputs, dim=-1).detach(), labels, pad_token=pad_token))  # 累积训练准确率
            train_loss.append(loss.item())  # 累积训练损失
            progress_bar.update(inputs.size(0))  # 更新进度条

        yield train_loss, sum(train_acc) / len(train_acc)

    progress_bar.close()  # 关闭进度条


def validate(
    model: MidiNet,
    dataset: MidiDataset,
    vocab_size: int,
    max_batch_size: int = 1,
    pad_token: int = 0,
    device: torch.device = None,
) -> tuple[int, int]:
    """
    对模型进行验证，返回平均损失和平均准确率。

    此函数遍历验证集，对模型进行前向传播，计算每个 batch 的交叉熵损失和准确率。
    所有 batch 的损失与准确率将被平均后返回，以衡量模型在整个验证集上的性能表现。

    Args:
        model: 需要验证的 MidiNet 模型。
        dataset: 用于验证的数据集。
        vocab_size: 词表大小，用于 reshape 输出。
        max_batch_size: 每个 batch 的最大大小。
        pad_token: padding 的 token 值，用于掩码处理和损失忽略。
        device: 计算设备（CPU 或 GPU）。

    Returns:
        平均损失和平均准确率，分别为 float 类型。

    Examples:
        >>> loss, acc = validate(model, val_dataset, vocab_size=128)
        >>> print(f"Validation Loss: {loss:.4f}, Accuracy: {acc:.2%}")
    """
    # 获取采样器，按 batch 划分数据
    sampler = MidiDatasetSampler(dataset, max_batch_size)

    # 获取验证数据加载器
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda x: sequence_collate_fn(x, pad_token=pad_token))

    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    # 初始化损失、准确率列表
    val_loss = []
    val_acc = []

    # 遍历整个验证集，不进行梯度计算
    for inputs, labels in dataloader:
        with torch.no_grad():
            # 将输入移动到计算设备
            inputs, labels = inputs.to(device), labels.to(device).view(-1)

            # 模型前向传播，得到输出并 reshape 成二维张量
            outputs = model(inputs, key_padding_mask=inputs == pad_token).view(-1, vocab_size)

            # 计算损失
            loss = criterion(outputs, labels)

            # 记录损失和准确率
            val_loss.append(loss.item())
            val_acc.append(compute_accuracy(torch.argmax(outputs, dim=-1).detach(), labels, pad_token=pad_token))

    # 返回平均损失和平均准确率
    return sum(val_loss) / len(val_loss), sum(val_acc) / len(val_acc)


def plot_training_process(metries: dict[str, list], img_path: pathlib.Path | str):
    """
    绘制训练过程中的损失、准确率曲线。

    Args:
        metries: 指标，包含
          - train_loss(list[list[float]]): 每个epoch的每一步的训练损失
          - train_accuracy(list[float]): 每个epoch的训练准确率平均值
          - val_accuracy(list[float]): 每个epoch的验证损失平均值
          - val_accuracy(list[float]): 每个epoch的验证准确率平均值
        img_path: 图形保存的文件路径，可以是字符串或Path对象。
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))  # 创建一个图形和一组坐标轴

    # 绘制训练过程中的损失曲线
    train_loss_x = [
        epoch + epoch_step / len(epoch_loss)
        for epoch, epoch_loss in enumerate(metries["train_loss"])
        for epoch_step in range(len(epoch_loss))
    ]
    ax1.plot(train_loss_x, [loss for epoch in metries["train_loss"] for loss in epoch], label="Train Loss", color="red")

    # 绘制验证过程中的损失曲线
    val_steps = list(range(1, len(metries["train_loss"]) + 1))
    ax1.plot(val_steps, metries["val_loss"], label="Validation Loss", marker=".", color="blue")

    # 设置第一个Y轴的标签
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epochs")

    # 创建第二个Y轴用于准确率
    ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴

    # 绘制训练过程中的准确率曲线
    ax2.plot(val_steps, metries["train_accuracy"], label="Train Accuracy", marker=".", color="green", linestyle="--")

    # 绘制验证过程中的准确率曲线
    ax2.plot([x + 0.5 for x in val_steps], metries["val_accuracy"], label="Validation Accuracy", marker=".", color="blue", linestyle="--")

    # 设置第二个Y轴的标签并转换为百分比
    ax2.set_ylabel("Accuracy")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{round(x * 100, 2)}%"))  # 格式化为百分比

    # 设置标题和图例
    ax1.set_title("Training Process")  # 设置标题
    ax1.legend(loc="upper left")  # 为损失曲线添加图例
    ax2.legend(loc="upper right")  # 为准确率曲线添加图例

    plt.tight_layout()  # 自动调整布局，防止标签重叠
    pathlib.Path(img_path).parent.mkdir(parents=True, exist_ok=True)  # 确保保存路径存在
    plt.savefig(img_path, dpi=300, bbox_inches="tight")  # 保存图形
    plt.show()  # 显示图形


def main():
    "训练 MIDI 模型并绘制训练过程中的损失、准确率曲线。"
    parser = argparse.ArgumentParser()
    parser.add_argument("num_epochs", type=int, help="训练的总轮数")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="加载和保存检查点的路径")
    parser.add_argument("-t", "--train-dataset", action="append", type=pathlib.Path, help="训练集文件路径（可多次指定以使用多个数据集）")
    parser.add_argument("-v", "--val-dataset", action="append", type=pathlib.Path, help="验证集文件路径（可多次指定以使用多个数据集）")
    parser.add_argument("-m", "--min-sequence-length", default=1024, type=int, help="最小序列长度，小于该长度的样本会被过滤掉")
    parser.add_argument("-b", "--max-batch-size", default=8 * 1024 ** 2, type=int, help="每个批次的序列长度的平方和上限")
    parser.add_argument("-l", "--learning-rate", default=0.01, type=float, help="学习率")
    parser.add_argument("-w", "--weight-decay", default=0.01, type=float, help="权重衰减系数")
    parser.add_argument("-d", "--d-model", default=512, type=int, help="嵌入向量的维度")
    parser.add_argument("-n", "--num-heads", default=8, type=int, help="多头注意力中的注意力头数量")
    parser.add_argument("-f", "--dim-feedforward", default=2048, type=int, help="前馈神经网络的隐藏层维度")
    parser.add_argument("-s", "--num-layers", default=6, type=int, help="模型 Transformer 编码器中的层数")
    parser.add_argument("-o", "--dropout", default=0.1, type=float, help="Dropout 概率，用于防止过拟合")
    args = parser.parse_args()

    # 训练集不能为空
    assert args.train_dataset, "训练集不能为空。"

    # 加载检查点
    tokenizer, model_state, optimizer_state, metries = load_checkpoint(args.ckpt_path, train=True)

    # 加载训练集
    train_dataset = MidiDataset(args.train_dataset, tokenizer, args.min_sequence_length)

    # 如果有的话，加载验证集
    if args.val_dataset:
        val_dataset = MidiDataset(args.val_dataset, tokenizer, args.min_sequence_length)

    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = MidiNet(len(tokenizer), args.d_model, args.num_heads, args.dim_feedforward, args.num_layers, dropout=args.dropout)

    # 加载模型状态
    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        metries = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    # 转移模型到设备
    model = model.to(device)

    # 检查是否使用多GPU
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # 使用 DataParallel 进行多GPU训练

    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 加载优化器状态
    try:
        optimizer.load_state_dict(optimizer_state)
    except (ValueError, KeyError):
        pass

    # 开始训练模型
    for train_loss, train_acc in train(model, train_dataset, optimizer, len(tokenizer), args.num_epochs, max_batch_size=args.max_batch_size, pad_token=tokenizer.pad_token_id, device=device):
        metries["train_loss"].append(train_loss)
        metries["train_accuracy"].append(train_acc)

        if args.val_dataset:
            val_loss, val_acc = validate(model, val_dataset, len(tokenizer), max_batch_size=args.max_batch_size, pad_token=tokenizer.pad_token_id, device=device)
            metries["val_loss"].append(val_loss)
            metries["val_accuracy"].append(val_acc)
        else:
            metries["val_loss"].append(float("nan"))
            metries["val_accuracy"].append(float("nan"))

    # 保存当前模型的检查点
    save_checkpoint(model, optimizer, metries, args.ckpt_path)

    # 绘制训练过程中的损失和准确率曲线
    plot_training_process(metries, "statistics.png")


if __name__ == "__main__":
    main()
