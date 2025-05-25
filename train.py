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
import json
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.nn import DataParallel, functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, SequentialSampler
from transformers import PreTrainedTokenizerFast

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
    from constants import DEFAULT_DIM_HEAD, DEFAULT_NUM_HEADS, DEFAULT_DIM_FEEDFORWARD, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT, DEFAULT_WEIGHT_DECAY, DEFAULT_LEARNING_RATE, DEFAULT_MIN_SEQUENCE_LENGTH
    from model import MidiNet
    from checkpoint import load_checkpoint, save_checkpoint
    from utils import midi_to_notes, notes_to_sheet, empty_cache
    from tokenizer import data_to_str


class MidiDataset(Dataset):
    """
    MIDI 数据集类，用于加载和处理 MIDI 文件，将其转化为模型可以使用的格式。

    功能说明:
    1. 从指定目录读取所有 MIDI 和 JSON 文件
    2. 将 MIDI 文件转换为音符序列
    3. 将音符序列分割为指定长度的训练样本
    4. 提供数据集大小和索引访问功能

    Attributes:
        all_music_data: 存储所有音乐文件的数据(原始字符串格式)
        music_sequences: 存储每个训练序列的元信息(文件索引、起始位置、长度)
        tokenizer: 用于将音乐数据转换为模型输入的分词器

    Args:
        midi_dirs: 包含 MIDI/JSON 文件的目录列表
        tokenizer: 用于音乐数据编码的分词器
        min_sequence_length: 训练序列的最小长度(按音符表示时的长度算，过短序列会被丢弃)
        max_sequence_length: 序列最大长度(按乐谱表示时的长度算，超长序列会被截断)
        show_progress: 是否显示加载进度条

    Examples:
        >>> dataset = MidiDataset(
        ...     midi_dirs=[pathlib.Path("data/midi")],
        ...     tokenizer=tokenizer,
        ...     min_sequence_length=64,
        ...     max_sequence_length=8964
        ... )
        >>> len(dataset)  # 获取训练样本数量
        198964
        >>> dataset[0]  # 获取第一个训练样本
    """

    def __init__(self, midi_dirs: list[pathlib.Path], tokenizer: PreTrainedTokenizerFast, min_sequence_length: int, max_sequence_length: int, show_progress: bool = True):
        self.music_sequences = []  # 存储每个训练序列的信息(乐谱分词表示)

        # 处理 MIDI 文件
        midi_files = [f for dir_path in midi_dirs for f in dir_path.glob("**/*.mid")]
        for filepath in tqdm(midi_files, desc="加载音乐数据集（原始 MIDI 文件）", delay=0.1, disable=not show_progress):
            # 读取并转化 MIDI 文件
            try:
                midi_file = mido.MidiFile(filepath, clip=True)
            except Exception:
                # 跳过有错误的 MIDI 文件
                continue

            # 提取音符
            notes = midi_to_notes(midi_file)

            # 转化为电子乐谱形式
            sheet, positions = notes_to_sheet(notes, max_length=max_sequence_length)

            # 跳过小于指定长度的 MIDI 文件
            if len(positions) < min_sequence_length:
                continue

            # 保存序列的内容
            self.music_sequences.append(tokenizer.encode(data_to_str(sheet)))

        # 处理 JSON 文件（更快的格式）
        json_files = [f for dir_path in midi_dirs for f in dir_path.glob("**/*.json")]

        for filepath in tqdm(json_files, desc="加载音乐数据集（优化的 JSON 文件）", delay=0.1, disable=not show_progress):
            # 读取文件
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            # 截断超长序列
            if len(data["data"]) > max_sequence_length:
                notes_end, sheet_end = max(
                    (i, position)
                    for i, position in enumerate(data["positions"])
                    if position < max_sequence_length
                )
                data["num_notes"] = notes_end
                data["data"] = data["data"][:sheet_end]

            # 跳过超短序列
            if data["num_notes"] < min_sequence_length:
                continue

            # 将当前 MIDI 文件的音符数据加入到 music_sequences 列表中
            self.music_sequences.append(tokenizer.encode(data["data"]))

    def __len__(self):
        return len(self.music_sequences)

    def __getitem__(self, index: int):
        sequence = self.music_sequences[index]
        return torch.tensor(sequence[:-1], dtype=int), torch.tensor(sequence[1:], dtype=int)


class MidiDatasetSampler(Sampler[list[int]]):
    """
    用于根据MIDI数据的长度动态分批的采样器类，适配最大token数限制。

    该采样器会根据给定的基础采样器顺序（如SequentialSampler或DistributedSampler），
    获取数据集中所有样本的长度信息，并按序列长度排序，确保每个批次中样本的
    总token数平方和不超过预设的max_batch_tokens。较长的序列可能会被跳过。

    每个批次被打乱顺序，以提升模型训练的泛化能力。

    Args:
        sampler: 一个返回索引的基础采样器（如RandomSampler）
        dataset: 一个包含MIDI音乐序列的数据集对象，要求具备music_sequences属性
        max_batch_tokens: 一个整数，指定每个批次中最大token平方和限制

    Returns:
        一个批次索引列表的迭代器，每个元素是一个样本索引组成的列表

    Examples:
        sampler = RandomSampler(dataset)
        batch_sampler = MidiDatasetSampler(sampler, dataset, max_batch_tokens=4096)
        for batch_indices in batch_sampler:
            batch = [dataset[i] for i in batch_indices]
    """

    def __init__(self, sampler: Sampler[int], dataset, max_batch_tokens: int):
        self.sampler = sampler
        self.dataset = dataset
        self.max_batch_tokens = max_batch_tokens

    def __iter__(self):
        # 获取基础采样器的所有索引
        indices = list(self.sampler)

        # 存储最终的所有批次
        all_batches = []
        current_batch = []
        current_token_sum = 0

        # 计算每个样本的序列长度 (index, length)，用于后续排序
        indexed_lengths = [
            (index, len(self.dataset.music_sequences[index])) for index in indices
        ]

        # 根据长度升序排序，长度相同时使用随机扰动避免固定排序
        sorted_indices = sorted(
            indexed_lengths,
            key=lambda x: (x[1], random.random())
        )

        for index, sequence_length in sorted_indices:
            estimated_cost = sequence_length ** 2

            # 如果单个样本就超过限制，则作为单独一个批次
            if estimated_cost > self.max_batch_tokens:
                all_batches.append([index])
                continue

            # 如果当前批次加上该样本会超出限制，则保存当前批次并开始新批次
            if current_token_sum + estimated_cost > self.max_batch_tokens:
                all_batches.append(current_batch)
                current_batch = []
                current_token_sum = 0

            # 添加样本到当前批次
            current_batch.append(index)
            current_token_sum += estimated_cost

        # 添加最后一个非空批次
        if current_batch:
            all_batches.append(current_batch)

        # 打乱批次顺序以增加训练随机性
        random.shuffle(all_batches)

        # 依次返回每个批次
        for batch in all_batches:
            yield batch


def sequence_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_token: int):
    "将多个样本合成统一长度的batch。"
    inputs, labels = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token)
    return inputs, labels


def train(
    model: MidiNet,
    dataloader: DataLoader,
    optimizer: optim.Adam,
    vocab_size: int,
    pad_token: int = 0,
    device: torch.device = None,
    show_progress: bool = True
) -> list[float]:
    """
    训练模型的函数。
    此函数进行一轮训练，逐步优化模型参数，输出训练损失。

    工作流程:
        1. 初始化数据加载器。
        2. 将模型移动到指定设备。
        3. 选择交叉熵损失函数。
        4. 使用进度条显示训练进度。
        5. 切换模型到训练模式。
        6. 对每个batch进行前向传播、计算损失、反向传播和更新参数。
        7. 累积损失。
        8. 返回这个epoch的平均训练损失。

    Args:
        model: 需要训练的神经网络模型。
        dataloader: 训练数据加载器。
        optimizer: 用于优化模型的优化器。
        vocab_size: 词汇表的大小，用于调整输出层的维度。
        pad_token: 填充token的标记，用于忽略计算损失。
        device: 指定训练的设备。
        show_progress: 是否显示加载进度条。

    Returns:
        训练损失。

    Examples:
        >>> tokenizer = PreTrainedTokenizerFast.from_pretrained("ckpt/tokenizer")
        >>> model = MidiNet(len(tokenizer), 768, 12, 2048, 12)
        >>> optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
        >>> criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        >>> train(model, dataloader, optimizer, len(tokenizer), tokenizer.pad_token_id)
        [1.9, 0.89, 0.6, 0.4]
    """
    # 清理缓存以释放内存
    empty_cache()

    # 将模型移动到设备
    model = model.to(device)

    # 设置模型为训练模式
    model.train()
    losses = []  # 初始化损失列表

    # 创建进度条，显示训练进度
    progress_bar = tqdm(total=len(dataloader.dataset), disable=not show_progress)
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).view(-1)
        optimizer.zero_grad()  # 清空梯度
        try:
            outputs = model(inputs, padding_mask=inputs == pad_token).view(-1, vocab_size)  # 前向传播并 reshape 成二维张量
            loss = F.cross_entropy(outputs, labels, ignore_index=pad_token)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
        except torch.cuda.OutOfMemoryError:
            empty_cache()
            continue

        # 更新进度条
        progress_bar.update(inputs.size(0))

        # 累积训练损失
        losses.append(loss.item())

    # 返回训练损失
    return losses


@torch.no_grad()
def validate(
    model: MidiNet,
    dataloader: DataLoader,
    vocab_size: int,
    pad_token: int = 0,
    device: torch.device = None,
    show_progress: bool = True
) -> list[float]:
    """
    对模型进行验证，返回平均损失和平均困惑度。

    此函数遍历验证集，对模型进行前向传播，计算每个 batch 的交叉熵损失。
    所有 batch 的损失将被平均后返回，以衡量模型在整个验证集上的性能表现。

    Args:
        model: 需要验证的 MidiNet 模型。
        dataloader: 验证数据加载器。
        vocab_size: 词表大小，用于 reshape 输出。
        pad_token: padding 的 token 值，用于掩码处理和损失忽略。
        device: 计算设备。
        show_progress: 是否显示加载进度条。

    Returns:
        验证损失。
    """
    # 清理缓存以释放内存
    empty_cache()

    # 初始化损失、困惑度列表
    losses = []

    # 遍历整个验证集，不进行梯度计算
    progress_bar = tqdm(total=len(dataloader.dataset), disable=not show_progress)
    for inputs, labels in dataloader:
        # 将输入移动到计算设备
        inputs, labels = inputs.to(device), labels.to(device).view(-1)

        # 模型前向传播，得到输出并 reshape 成二维张量
        outputs = model(inputs, padding_mask=inputs == pad_token).view(-1, vocab_size)

        # 计算并记录损失
        losses.extend(F.cross_entropy(outputs, labels, ignore_index=pad_token, reduction="none").tolist())

        # 更新进度条
        progress_bar.update(inputs.size(0))

    # 返回损失
    return losses


def plot_training_process(metrics: dict[str, list], img_path: pathlib.Path | str):
    """
    绘制损失变化过程。训练损失使用红线，验证损失用蓝色点线。
    为每种损失分别绘制置信区间。

    Args:
        metrics: 指标，包含
          - train_loss(list[list[float]]): 每个epoch的的训练损失平均值
          - val_loss(list[list[float]]): 每个epoch的每一个验证样本的损失
        img_path: 图形保存的文件路径，可以是字符串或Path对象。
    """
    # 创建图形和坐标轴
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 计算验证点的x坐标（每个epoch的起始位置）
    val_iteration_points = []  # 存储每个epoch的起始迭代次数
    current_iteration = 0  # 当前累计的迭代次数
    for epoch in metrics["train_loss"]:
        val_iteration_points.append(current_iteration)
        current_iteration += len(epoch)  # 累加当前epoch的迭代次数

    # 计算训练损失曲线的x坐标偏移量（0.5个epoch的位置）
    train_x_offset = sum(len(epoch_losses) for epoch_losses in metrics["train_loss"]) / len(metrics["train_loss"]) / 2
    train_x = [x + train_x_offset for x in val_iteration_points]

    # 计算训练损失的统计量
    train_loss_avg = [np.mean(epoch_losses) for epoch_losses in metrics["train_loss"]]
    train_loss_upper = [train_loss_avg[i] + np.std(epoch_losses) for i, epoch_losses in enumerate(metrics["train_loss"])]
    train_loss_lower = [train_loss_avg[i] - np.std(epoch_losses) for i, epoch_losses in enumerate(metrics["train_loss"])]

    # 绘制训练损失曲线和标准差区间
    ax1.plot(train_x, train_loss_avg, label="Train Loss", color="red", linestyle="-", marker=".")
    ax1.fill_between(train_x, train_loss_upper, train_loss_lower, color="red", alpha=0.2)

    # 计算验证损失的统计量
    val_loss_avg = [np.mean(epoch_losses) for epoch_losses in metrics["val_loss"]]
    val_loss_upper = [val_loss_avg[i] + np.std(epoch_losses) for i, epoch_losses in enumerate(metrics["val_loss"])]
    val_loss_lower = [val_loss_avg[i] - np.std(epoch_losses) for i, epoch_losses in enumerate(metrics["val_loss"])]

    # 绘制验证损失曲线和标准差区间
    ax1.plot(val_iteration_points, val_loss_avg, label="Validation Loss", color="blue", linestyle="-", marker=".")
    ax1.fill_between(val_iteration_points, val_loss_upper, val_loss_lower, color="blue", alpha=0.2)

    # 设置坐标轴标签和标题
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Iteration")
    ax1.set_title("Training Process with Dual Confidence Intervals")

    # 添加图例和网格
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 保存并展示图形
    plt.tight_layout()
    pathlib.Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="训练 MIDI 模型并绘制训练过程中的损失、困惑度曲线。")

    # 添加必须参数
    parser.add_argument("num_epochs", type=int, help="训练的总轮数")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="加载和保存检查点的路径")
    parser.add_argument("-t", "--train-dataset", action="append", type=pathlib.Path, required=True, help="训练集文件路径（可多次指定以使用多个数据集）")

    # 添加可选参数
    parser.add_argument("-v", "--val-dataset", action="append", type=pathlib.Path, help="验证集文件路径（可多次指定以使用多个数据集）")
    parser.add_argument("-m", "--min-sequence-length", default=DEFAULT_MIN_SEQUENCE_LENGTH, type=int, help="最小序列长度，小于该长度的样本不会被训练")
    parser.add_argument("-e", "--max-sequence-length", default=2 ** 17, type=int, help="最大序列长度，大于该长度的样本将被截断")
    parser.add_argument("-b", "--max-batch-size", default=1536 ** 2, type=int, help="每个批次的序列长度的平方和上限")
    parser.add_argument("-l", "--learning-rate", default=DEFAULT_LEARNING_RATE, type=float, help="学习率")
    parser.add_argument("-w", "--weight-decay", default=DEFAULT_WEIGHT_DECAY, type=float, help="权重衰减系数")
    parser.add_argument("-n", "--num-heads", default=DEFAULT_NUM_HEADS, type=int, help="多头注意力中的注意力头数量")
    parser.add_argument("-d", "--dim-head", default=DEFAULT_DIM_HEAD, type=int, help="多头注意力中的注意力头的维度")
    parser.add_argument("-f", "--dim-feedforward", default=DEFAULT_DIM_FEEDFORWARD, type=int, help="前馈神经网络的隐藏层维度")
    parser.add_argument("-s", "--num-layers", default=DEFAULT_NUM_LAYERS, type=int, help="模型 Transformer 编码器中的层数")
    parser.add_argument("-o", "--dropout", default=DEFAULT_DROPOUT, type=float, help="Dropout 概率，用于防止过拟合")

    # 解析命令行参数并返回
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 清理缓存以释放内存
    empty_cache()

    # 加载检查点
    tokenizer, model_state, optimizer_state, metrics = load_checkpoint(args.ckpt_path, train=True)

    # 加载训练集并创建数据加载器
    train_dataset = MidiDataset(args.train_dataset, tokenizer, args.min_sequence_length, args.max_sequence_length)
    train_loader = DataLoader(train_dataset, batch_sampler=MidiDatasetSampler(SequentialSampler(train_dataset), train_dataset, args.max_batch_size), collate_fn=lambda x: sequence_collate_fn(x, pad_token=tokenizer.pad_token_id))

    # 如果有的话，加载验证集并创建数据加载器
    if args.val_dataset:
        val_dataset = MidiDataset(args.val_dataset, tokenizer, args.min_sequence_length, args.max_sequence_length)
        val_loader = DataLoader(val_dataset, batch_sampler=MidiDatasetSampler(SequentialSampler(val_dataset), val_dataset, args.max_batch_size), collate_fn=lambda x: sequence_collate_fn(x, pad_token=tokenizer.pad_token_id))

    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = MidiNet(len(tokenizer), args.num_heads, args.dim_head, args.dim_feedforward, args.num_layers, dropout=args.dropout)

    # 加载模型状态
    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        metrics = {"train_loss": [], "val_loss": []}

    # 转移模型到设备
    model = model.to(device)

    # 如果存在多个GPU，则使用 DataParallel 进行训练
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 加载优化器状态
    try:
        optimizer.load_state_dict(optimizer_state)
    except (ValueError, KeyError):
        pass

    # 开始训练模型
    for epoch in range(args.num_epochs):
        # 训练并添加损失到指标
        train_loss = train(model, train_loader, optimizer, len(tokenizer), tokenizer.pad_token_id, device)
        metrics["train_loss"].append(train_loss)

        # 如果指定了验证集，就进行验证，否则跳过验证并设置验证损失为 NaN
        if args.val_dataset:
            val_loss = validate(model, val_loader, len(tokenizer), tokenizer.pad_token_id, device)
        else:
            val_loss = [float("nan")]

        # 添加验证损失到指标
        metrics["val_loss"].append(val_loss)

    # 保存当前模型的检查点
    save_checkpoint(model, optimizer, metrics, args.ckpt_path)

    # 绘制训练过程中的损失和困惑度曲线
    plot_training_process(metrics, "statistics.png")


if __name__ == "__main__":
    main()
