"MIDI 音乐生成模型训练模块"

# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import pathlib
import random
import argparse
import json
import os
import tempfile
from multiprocessing import cpu_count
from typing import Optional, Iterator
import mido
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch import nn, optim, distributed as dist, multiprocessing as mp
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import PreTrainedTokenizerFast
from constants import DEFAULT_DIM_HEAD, DEFAULT_NUM_HEADS, DEFAULT_DIM_FEEDFORWARD, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT, DEFAULT_WEIGHT_DECAY, DEFAULT_LEARNING_RATE, DEFAULT_MIN_SEQUENCE_LENGTH
from model import MidiNet, MidiNetConfig
from checkpoint import load_checkpoint_train, save_checkpoint
from utils import midi_to_notes, notes_to_sheet, empty_cache
from tokenizer import data_to_str

# 解除线程数量限制
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count())
torch.set_num_threads(cpu_count())


class MidiDataset(Dataset):
    """
    MIDI 数据集类，用于加载和处理 MIDI 文件，将其转化为模型可以使用的格式。

    功能说明:
    1. 从指定目录读取所有 MIDI 和 JSON 文件
    2. 将 MIDI 文件转换为音符序列
    3. 将音符序列分割为指定长度的训练样本
    4. 提供数据集大小和索引访问功能

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
        midi_files = sorted([f for dir_path in midi_dirs for f in dir_path.glob("**/*.mid")], key=lambda file: file.name)
        for filepath in tqdm(midi_files, desc="加载音乐数据集（原始 MIDI 文件）", delay=0.1, disable=not show_progress):
            # 读取并转化 MIDI 文件
            try:
                midi_file = mido.MidiFile(filepath, clip=True)
            except (ValueError, EOFError, OSError):
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
        json_files = sorted([f for dir_path in midi_dirs for f in dir_path.glob("**/*.json")], key=lambda file: file.name)

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
        sequence = torch.tensor(self.music_sequences[index], dtype=int)
        return sequence[:-1], sequence[1:]


class MidiDatasetSampler(Sampler[list[int]]):
    """
    用于 MIDI 数据集的分批采样器，根据序列长度进行动态批处理。
    
    该采样器会:
    1. 根据序列长度对样本进行排序
    2. 动态创建批次，确保每个批次的token总数不超过max_batch_tokens
    3. 支持分布式训练环境下的数据分配
    4. 每个epoch都会重新打乱数据顺序

    Attributes:
        max_batch_tokens: 单个批次允许的最大token数量
        rank: 当前进程的rank编号(分布式训练用)
        world_size: 总进程数(分布式训练用)
        seed: 随机种子
        batches: 当前rank分配到的批次列表
        total_tokens: 当前rank分配到的总token数

    Examples:
        >>> dataset = MidiDataset([pathlib.Path("data/")], tokenizer, min_sequence_length=64, max_sequence_length=8964)
        >>> sampler = MidiDatasetSampler(dataset, max_batch_tokens=4096)
        >>> for batch in sampler:
        ...     print(batch)  # [19, 89, 64]
    """

    def __init__(self, dataset: MidiDataset, max_batch_tokens: int, rank: int = 0, world_size: int = 1, seed: int = 0):
        super().__init__()
        self.max_batch_tokens = max_batch_tokens
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.batches = []
        self.total_tokens = 0

        # 预计算所有样本的索引和长度
        self.index_and_lengths = [(idx, len(dataset.music_sequences[idx])) for idx in range(len(dataset))]
        self.index_to_length = dict(self.index_and_lengths)

    def set_epoch(self, epoch: int) -> None:
        """
        设置当前epoch并重新生成批次
        
        每个epoch开始时调用，用于:
        1. 根据新epoch重新打乱数据顺序
        2. 重新分配批次
        3. 确保分布式环境下各rank数据对齐
        """
        generator = random.Random(self.seed + epoch)

        # 按长度排序，加入随机因子避免固定排序
        sorted_pairs = sorted(self.index_and_lengths, key=lambda pair: (pair[1], generator.random()))

        batches_with_tokens: list[tuple[list[int], int]] = []
        current_batch: list[int] = []

        for idx, seq_len in sorted_pairs:
            token_len = seq_len - 1  # 输入序列比原序列少1

            # 处理超长序列
            if token_len > self.max_batch_tokens:
                batches_with_tokens.append(([idx], token_len))
                continue

            # 计算当前批次加入新样本后的token总数
            estimated_tokens = (len(current_batch) + 1) * token_len
            if estimated_tokens > self.max_batch_tokens:
                # 当前批次中最长序列决定了该批次的token总数
                longest_in_batch = self.index_to_length[current_batch[-1]] - 1
                batch_tokens = longest_in_batch * len(current_batch)
                batches_with_tokens.append((current_batch, batch_tokens))
                current_batch = []

            current_batch.append(idx)

        # 添加最后一个批次
        if current_batch:
            longest_in_batch = self.index_to_length[current_batch[-1]] - 1
            batch_tokens = longest_in_batch * len(current_batch)
            batches_with_tokens.append((current_batch, batch_tokens))

        # 确保批次数是world_size的倍数
        if len(batches_with_tokens) % self.world_size:
            needed = self.world_size - (len(batches_with_tokens) % self.world_size)
            # 随机复制现有批次来补全
            for _ in range(needed):
                batch_idx = generator.randint(0, len(batches_with_tokens) - 1)
                batches_with_tokens.insert(batch_idx, batches_with_tokens[batch_idx])

        # 批次倒序，用于快速检测训练的问题
        batches_with_tokens.reverse()

        # 分配当前rank的批次
        self.batches = batches_with_tokens[self.rank::self.world_size]
        self.total_tokens = sum(tokens for _, tokens in self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        yield from (batch for batch, _ in self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def sequence_collate_fn(batch, pad_token=0):
    """
    将一个批次的变长输入和标签序列整理为统一长度的张量（自动填充pad_token）。

    参数:
        batch: 每个元素是一个元组，包含输入张量和标签张量。
        pad_token: 用于填充的token值。

    返回:
        填充后的输入序列和标签序列张量。
    """
    # 将多个样本合成统一长度的batch。
    inputs, labels = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token)
    return inputs, labels


def train(
    model: MidiNet,
    dataloader: DataLoader,
    optimizer: optim.AdamW,
    vocab_size: int,
    pad_token: int = 0,
    device: Optional[torch.device] = None,
    show_progress: bool = True
) -> tuple[list[float], list[tuple[int, int]]]:
    """
    训练模型的函数。
    此函数进行一轮训练，逐步优化模型参数，输出训练损失和触发OOM的输入形状。

    工作流程:
        1. 初始化数据加载器。
        2. 将模型移动到指定设备。
        3. 选择交叉熵损失函数。
        4. 使用进度条显示训练进度。
        5. 切换模型到训练模式。
        6. 对每个batch进行前向传播、计算损失、反向传播和更新参数。
        7. 累积损失。
        8. 返回这个epoch的训练损失和触发OOM的输入形状。

    Args:
        model: 需要训练的神经网络模型。
        dataloader: 训练数据加载器。
        optimizer: 用于优化模型的优化器。
        vocab_size: 词汇表的大小，用于调整输出层的维度。
        pad_token: 填充token的标记，用于忽略计算损失。
        device: 指定训练的设备。
        show_progress: 是否显示进度条。

    Notes:
        - 默认使用混合精度训练（FP16 +梯度缩放）。
        - 遇到OOM时会自动跳过当前batch并记录输入形状。

    Returns:
        - losses: 每个batch的训练损失列表。
        - oom_shapes: 触发CUDA OOM（内存不足）的输入形状列表，格式: [(batch_size, seq_len), ...]

    Examples:
        >>> tokenizer = PreTrainedTokenizerFast.from_pretrained("ckpt/tokenizer")
        >>> model = MidiNet(MidiNetConfig(len(tokenizer), 8, 64, 2048, 12))
        >>> optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
        >>> train(model, dataloader, optimizer, len(tokenizer), tokenizer.pad_token_id)
        ([1.9, 0.89, 0.6, 0.4], [])
    """
    # 清理缓存以释放内存
    empty_cache()

    # 将模型移动到设备
    model = model.to(device)

    # 设置模型为训练模式
    model.train()

    # 用于梯度缩放
    scaler = GradScaler()

    # 创建进度条，显示训练进度
    dataloader_iter = iter(dataloader)
    progress_bar = tqdm(total=dataloader.batch_sampler.total_tokens, disable=not show_progress)

    # 初始化损失列表、触发OOM的输入形状列表
    losses = []
    oom_shapes = []  # shape: [batch_size, seq_len]

    # 提前确定 device_type，避免多次判断
    device_type = device.type if device is not None else "cpu"

    # 遍历整个训练集
    for inputs, labels in dataloader_iter:
        outputs = loss = None
        inputs, labels = inputs.to(device), labels.to(device)
        progress_n = inputs.size(0) * inputs.size(1)  # 进度条更新的步数
        optimizer.zero_grad()  # 清空梯度

        try:
            # 使用半精度节省显存
            with autocast(device_type, dtype=torch.float16):
                outputs = model(inputs, padding_mask=inputs == pad_token).view(-1, vocab_size)  # 前向传播并 reshape 成二维张量
                loss = F.cross_entropy(outputs, labels.view(-1), ignore_index=pad_token)  # 计算损失

            scaler.scale(loss).backward()  # 反向传播
            scaler.step(optimizer)  # 更新模型参数
            scaler.update()  # 调整缩放因子

            losses.append(loss.item())  # 累积训练损失
            progress_bar.set_postfix(loss=loss.item())  # 更新进度条
        except torch.OutOfMemoryError:
            # 记录OOM时的输入形状
            oom_shapes.append(tuple(inputs.shape))

            # 保持 DDP 同步
            optimizer.zero_grad()
            with autocast(device_type, dtype=torch.float16):
                outputs = model(torch.zeros((1, 1), dtype=int, device=device)).view(-1, vocab_size)
                loss = F.cross_entropy(outputs, torch.zeros(outputs.size(0), dtype=int, device=device), ignore_index=pad_token)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 消除引用，方便垃圾回收
            inputs = labels = None

            # 清理缓存以释放内存
            empty_cache()

        # 更新进度条
        progress_bar.update(progress_n)

    # 关闭进度条
    progress_bar.close()

    # 返回训练损失和OOM输入形状
    return losses, oom_shapes


@torch.inference_mode()
def validate(
    model: MidiNet,
    dataloader: DataLoader,
    vocab_size: int,
    pad_token: int = 0,
    device: Optional[torch.device] = None,
    show_progress: bool = True
) -> list[float]:
    """
    对模型进行验证，返回每个 batch 的损失。

    此函数遍历验证集，对模型进行前向传播，计算每个 batch 的交叉熵损失。
    返回所有 batch 的损失列表，以衡量模型在整个验证集上的性能表现。

    Args:
        model: 需要验证的 MidiNet 模型。
        dataloader: 验证数据加载器。
        vocab_size: 词表大小，用于 reshape 输出。
        pad_token: padding 的 token 值，用于掩码处理和损失忽略。
        device: 计算设备。
        show_progress: 是否显示进度条。

    Returns:
        验证损失列表。
    """
    # 清理缓存以释放内存
    empty_cache()

    # 设置模型为评估模式
    model.eval()

    # 创建进度条，显示验证进度
    dataloader_iter = iter(dataloader)
    progress_bar = tqdm(total=dataloader.batch_sampler.total_tokens, disable=not show_progress)

    # 初始化损失列表
    losses = []

    # 遍历整个验证集
    for inputs, labels in dataloader_iter:
        # 将输入移动到计算设备
        inputs, labels = inputs.to(device), labels.to(device)

        # 使用半精度节省显存
        with autocast(device.type if device else "cpu", dtype=torch.float16):
            # 模型前向传播，得到输出并 reshape 成二维张量
            outputs = model(inputs, padding_mask=inputs == pad_token).view(-1, vocab_size)

            # 计算并记录损失
            loss = F.cross_entropy(outputs, labels.view(-1), ignore_index=pad_token).item()
            losses.append(loss)

        # 更新进度条
        progress_bar.update(inputs.size(0) * inputs.size(1))

    # 关闭进度条
    progress_bar.close()

    # 返回损失
    return losses


def plot_training_process(metrics: dict[str, list], img_path: pathlib.Path | str):
    """
    绘制损失变化过程。训练损失使用红线，验证损失用蓝色点线。
    为每种损失分别绘制置信区间。

    Args:
        metrics: 指标，包含
          - train_loss(list[dict[str, Any]]): 每个epoch的训练损失数量、平均值、标准差
          - val_loss(dict[str, Any]): 每个epoch的验证损失平均值、标准差
        img_path: 图形保存的文件路径，可以是字符串或Path对象。

    Example:
        ```
        metrics = {
            "train_loss": [
                {"mean": 1.2, "std_dev": 0.1, "count": 100},
                {"mean": 1.0, "std_dev": 0.08, "count": 100},
            ],
            "val_loss": [
                {"mean": 1.1, "std_dev": 0.09},
                {"mean": 0.95, "std_dev": 0.07},
            ]
        }
        ```
    """
    # 创建图形和坐标轴
    _, ax = plt.subplots(figsize=(10, 6))

    # 计算验证点的x坐标（每个epoch的起始位置）
    current_iteration = metrics["train_loss"][0]["count"]  # 当前累计的迭代次数
    val_iteration_points = [current_iteration]  # 存储每个epoch的起始迭代次数
    for epoch in metrics["train_loss"][1:]:
        current_iteration += epoch["count"]  # 累加当前epoch的迭代次数
        val_iteration_points.append(current_iteration)

    # 计算训练损失曲线的x坐标（偏移半个epoch）
    # 这里将每个训练损失点放在其对应 epoch 区间的中间位置（即当前验证点左移半个 epoch），
    # 这样可以更直观地反映该 epoch 内的训练损失均值与验证损失的时间关系。
    train_x = [val_iteration_points[i] - epoch["count"] / 2 for i, epoch in enumerate(metrics["train_loss"])]

    # 绘制训练损失曲线和标准差区间
    ax.plot(train_x, [epoch["mean"] for epoch in metrics["train_loss"]], label="Train Loss", color="red", linestyle="-", marker=".")
    train_upper = [epoch["mean"] + epoch["std_dev"] for epoch in metrics["train_loss"]]
    train_lower = [epoch["mean"] - epoch["std_dev"] for epoch in metrics["train_loss"]]
    ax.fill_between(train_x, train_upper, train_lower, color="red", alpha=0.2)

    ax.plot(val_iteration_points, [epoch["mean"] for epoch in metrics["val_loss"]], label="Validation Loss", color="blue", linestyle="-", marker=".")
    val_upper = [epoch["mean"] + epoch["std_dev"] for epoch in metrics["val_loss"]]
    val_lower = [epoch["mean"] - epoch["std_dev"] for epoch in metrics["val_loss"]]
    ax.fill_between(val_iteration_points, val_upper, val_lower, color="blue", alpha=0.2)

    # 设置X轴为整数刻度
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 设置坐标轴标签和标题
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    ax.set_title("Training Process")

    # 添加图例和网格
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 保存图形
    plt.tight_layout()
    pathlib.Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    """
    解析训练 MIDI 模型的命令行参数。

    Args:
        num_epochs (int): 训练总轮数（必填）。
        ckpt_path (pathlib.Path): 检查点的加载和保存路径（必填）。
        -t/--train-dataset (pathlib.Path, 多次): 训练集文件路径（必填，可多次指定）。
        -v/--val-dataset (pathlib.Path, 多次): 验证集文件路径（可选，可多次指定）。
        -m/--min-sequence-length (int): 最小序列长度，短于该长度的样本不会用于训练。
        -e/--max-sequence-length (int): 最大序列长度，长于该长度的样本会被截断。
        -b/--train-max-batch-tokens (int): 训练时每个批次序列长度之和的上限。
        -q/--val-max-batch-tokens (int): 验证时每个批次序列长度之和的上限。
        -l/--learning-rate (float): 学习率。
        -w/--weight-decay (float): 权重衰减系数。
        -n/--num-heads (int): 多头注意力的头数。
        -d/--dim-head (int): 每个注意力头的维度。
        -f/--dim-feedforward (int): 前馈网络隐藏层维度。
        -s/--num-layers (int): Transformer 编码器层数。
        -o/--dropout (float): Dropout 概率。
        -u/--seed (int): 随机种子，保证可复现性。

    Returns:
        解析后的命令行参数对象。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="训练 MIDI 模型并绘制训练过程中的损失、困惑度曲线。")

    # 添加必须参数
    parser.add_argument("num_epochs", type=int, help="训练的总轮数")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="加载和保存检查点的路径")
    parser.add_argument("-t", "--train-dataset", action="append", type=pathlib.Path, required=True, help="训练集文件路径（可多次指定以使用多个数据集）")

    # 添加可选参数
    parser.add_argument("-v", "--val-dataset", action="append", type=pathlib.Path, help="验证集文件路径（可多次指定以使用多个数据集）")
    parser.add_argument("-m", "--min-sequence-length", default=DEFAULT_MIN_SEQUENCE_LENGTH, type=int, help="最小序列长度，小于该长度的样本不会被训练")
    parser.add_argument("-e", "--max-sequence-length", default=16384, type=int, help="最大序列长度，大于该长度的样本将被截断，默认为 %(default)s")
    parser.add_argument("-b", "--train-max-batch-tokens", default=16384, type=int, help="训练时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-q", "--val-max-batch-tokens", default=32678, type=int, help="验证时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-l", "--learning-rate", default=DEFAULT_LEARNING_RATE, type=float, help="学习率，默认为 %(default)s")
    parser.add_argument("-w", "--weight-decay", default=DEFAULT_WEIGHT_DECAY, type=float, help="权重衰减系数，默认为 %(default)s")
    parser.add_argument("-n", "--num-heads", default=DEFAULT_NUM_HEADS, type=int, help="多头注意力中的注意力头数量，默认为 %(default)s")
    parser.add_argument("-d", "--dim-head", default=DEFAULT_DIM_HEAD, type=int, help="多头注意力中的注意力头的维度，默认为 %(default)s")
    parser.add_argument("-f", "--dim-feedforward", default=DEFAULT_DIM_FEEDFORWARD, type=int, help="前馈神经网络的隐藏层维度，默认为 %(default)s")
    parser.add_argument("-s", "--num-layers", default=DEFAULT_NUM_LAYERS, type=int, help="模型 Transformer 编码器中的层数，默认为 %(default)s")
    parser.add_argument("-o", "--dropout", default=DEFAULT_DROPOUT, type=float, help="Dropout 概率，用于防止过拟合，默认为 %(default)s")
    parser.add_argument("-u", "--seed", default=8964, type=int, help="训练的种子，保证训练过程可复现，默认为 %(default)s")

    # 解析命令行参数并返回
    return parser.parse_args()


def set_seed(seed: int):
    """
    设置所有随机源的种子以确保实验可复现性。

    工作流程:
    1. 设置Python内置random模块的种子
    2. 设置NumPy的随机种子
    3. 设置PyTorch的CPU和GPU随机种子
    4. 配置CuDNN使用确定性算法并关闭benchmark模式

    Args:
        seed: 要设置的随机种子值

    Examples:
        >>> set_seed(8964)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _mp_fn(rank: int, world_size: int, args: argparse.Namespace):
    """
    分布式训练主函数，用于 torch.multiprocessing.spawn 启动多进程训练。

    工作流程：
    1. 初始化分布式训练环境（如果world_size > 1）
    2. 加载模型、数据集和优化器
    3. 使用DistributedDataParallel包装模型
    4. 执行训练和验证循环
    5. 收集并汇总各进程的指标
    6. 保存模型检查点和训练统计信息
    7. 清理分布式训练环境

    Args:
        rank: 当前进程的排名（0为主进程）
        world_size: 总进程数（通常等于GPU数量）
        args: 包含所有训练配置的参数对象

    Examples:
        >>> torch.multiprocessing.spawn(_mp_fn, args=(world_size, args), nprocs=world_size)
    """
    # 初始化分布式训练组
    if world_size > 1:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # 设置当前进程的设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # 清理缓存以释放内存
    empty_cache()

    # 加载训练检查点（包括 tokenizer、模型、优化器状态、指标）
    tokenizer, model_state_dict, optimizer_state_dict, metrics = load_checkpoint_train(args.ckpt_path)

    # 加载训练数据集及分布式采样器
    train_dataset = MidiDataset(args.train_dataset, tokenizer, args.min_sequence_length, args.max_sequence_length, show_progress=rank == 0)
    train_sampler = MidiDatasetSampler(train_dataset, args.train_max_batch_tokens, rank, world_size, args.seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=lambda x: sequence_collate_fn(x, pad_token=tokenizer.pad_token_id))

    # 如果存在验证集，加载验证数据集
    if args.val_dataset:
        val_dataset = MidiDataset(args.val_dataset, tokenizer, args.min_sequence_length, args.max_sequence_length, show_progress=rank == 0)
        val_sampler = MidiDatasetSampler(val_dataset, args.val_max_batch_tokens, rank, world_size, args.seed)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=lambda x: sequence_collate_fn(x, pad_token=tokenizer.pad_token_id))

    # 确保使用确定性算法
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    set_seed(args.seed)

    # 初始化模型结构
    model = MidiNet(MidiNetConfig(len(tokenizer), args.num_heads, args.dim_head, args.dim_feedforward, args.num_layers), dropout=args.dropout)

    # 加载模型权重
    if model_state_dict:
        model.load_state_dict(model_state_dict)

    # 转移模型到设备
    model = model.to(device)

    # 用 DDP 包装模型
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 创建优化器
    optimizer = optim.AdamW(model.parameters())

    # 加载优化器状态
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    # 设置学习率、权重衰减系数
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
        group["weight_decay"] = args.weight_decay

    # 开始训练
    for epoch in range(args.num_epochs):
        # 设置每轮不同的随机种子（确保数据shuffle不同但可复现）
        current_epoch = len(metrics["train_loss"]) + epoch
        set_seed(args.seed + rank + current_epoch)

        # 训练一轮模型
        train_sampler.set_epoch(current_epoch)
        train_loss, oom_shapes = train(model, train_loader, optimizer, len(tokenizer), tokenizer.pad_token_id, device, show_progress=rank == 0)

        # 如果指定了验证集，就进行验证，否则跳过验证并设置验证损失为 NaN
        if args.val_dataset:
            val_sampler.set_epoch(current_epoch)
            val_loss = validate(model, val_loader, len(tokenizer), tokenizer.pad_token_id, device, show_progress=rank == 0)
        else:
            val_loss = [float("nan")]

        # 主进程放入自己的损失
        if rank == 0:
            all_train_loss = [train_loss]
            all_val_loss = [val_loss]

        # 多进程时汇集所有进程的损失
        if world_size > 1:
            # 获取系统临时目录路径
            temp_dir = pathlib.Path(tempfile.gettempdir())

            # 非主进程将本地损失写入以 rank 命名的文件
            if rank != 0:
                with open(temp_dir / f"rank{rank}.dat", "w", encoding="utf-8") as f:
                    json.dump({"train_loss": train_loss, "val_loss": val_loss}, f)  # 序列化训练、验证损失

            # 所有进程同步，确保写入操作完成再进行下一步
            dist.barrier()

            # 主进程读取所有其他进程写入的数据文件
            if rank == 0:
                for other_rank in range(1, world_size):
                    data_file = temp_dir / f"rank{other_rank}.dat"
                    with open(data_file, encoding="utf-8") as f:
                        content = json.load(f)
                        all_train_loss.append(content["train_loss"])
                        all_val_loss.append(content["val_loss"])

                    # 删除临时数据文件
                    data_file.unlink(missing_ok=True)

            # 所有进程同步，确保主进程读取完成再进行下一步
            dist.barrier()

        # 计算并添加损失平均值和标准差到指标
        if rank == 0:
            all_train_loss = np.array([loss for rank_loss in all_train_loss for loss in rank_loss])
            all_val_loss = np.array([loss for rank_loss in all_val_loss for loss in rank_loss])
            metrics["train_loss"].append({"mean": all_train_loss.mean(), "std_dev": all_train_loss.std(), "count": len(all_train_loss)})
            metrics["val_loss"].append({"mean": all_val_loss.mean(), "std_dev": all_val_loss.std()})

    # 收集所有进程中的 OOM 张量形状
    if rank == 0:
        all_oom_shapes = [oom_shapes]

    if world_size > 1:
        # 获取系统临时目录路径
        temp_dir = pathlib.Path(tempfile.gettempdir())

        # 非主进程将本地 OOM 张量形状写入以 rank 命名的文件
        if rank != 0:
            with open(temp_dir / f"rank{rank}.dat", "w", encoding="utf-8") as f:
                f.write(json.dumps(oom_shapes))

        # 所有进程同步，确保写入操作完成再进行下一步
        dist.barrier()

        # 主进程读取所有其他进程写入的数据文件
        if rank == 0:
            for other_rank in range(1, world_size):
                data_file = temp_dir / f"rank{other_rank}.dat"
                with open(data_file, encoding="utf-8") as f:
                    # 读取完整内容，并加入到列表中
                    all_oom_shapes.append(json.load(f))

                # 删除临时数据文件
                data_file.unlink(missing_ok=True)

    # 主进程保存模型、统计信息和 OOM 形状
    if rank == 0:
        # 保存最后一次训练时使内存爆炸的张量的形状
        all_oom_shapes = [oom_shape for rank_oom_shapes in all_oom_shapes for oom_shape in rank_oom_shapes]
        if all_oom_shapes:
            with open("oom_shapes.txt", "w", encoding="utf-8") as f:
                f.write("Shape (e.g: Batch Size x Sequence Length)\n")
                f.write("\n".join(f"{batch_size} x {sequence_length}" for batch_size, sequence_length in all_oom_shapes))

        # 保存当前模型的检查点
        save_checkpoint((model.module if world_size > 1 else model).cpu().state_dict(), optimizer.state_dict(), metrics, args.ckpt_path)

        # 绘制训练过程中的损失曲线
        plot_training_process(metrics, args.ckpt_path / "statistics.png")

    # 释放资源
    if world_size > 1:
        dist.destroy_process_group()


def main():
    """
    主函数，负责调配多进程训练。

    支持三种运行模式:
    1. 多 GPU：自动检测多块 GPU，使用分布式数据并行（DDP）加速训练。
    2. 单 GPU：仅检测到一块 GPU 时，使用单进程在该 GPU 上训练。
    3. CPU：无可用 GPU 时，自动切换为 CPU 训练模式。
    """
    # 解析命令行参数
    args = parse_args()

    # 如果有多 GPU，使用 DDP 加速训练
    world_size = max(torch.cuda.device_count(), 1)
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        # 随机选择一个动态/私有端口（49152–65535），以减少与常用端口冲突的概率
        os.environ["MASTER_PORT"] = str(random.randint(2 ** 15, 2 ** 16 - 1))
    else:
        # 单进程（单 GPU 或 CPU）模式
        _mp_fn(0, 1, args)
        _mp_fn(0, 1, args)


if __name__ == "__main__":
    main()
