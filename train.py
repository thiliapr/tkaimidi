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
import itertools
import os
from multiprocessing import cpu_count
import re
from typing import Optional, Iterator
import mido
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim, distributed as dist, multiprocessing as mp
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from transformers import PreTrainedTokenizerFast

# 解除线程数量限制
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count())
torch.set_num_threads(cpu_count())

# 根据是否在 Jupyter 环境下导入不同库
if "get_ipython" in globals():
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
    from constants import DEFAULT_DIM_HEAD, DEFAULT_NUM_HEADS, DEFAULT_DIM_FEEDFORWARD, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT, DEFAULT_WEIGHT_DECAY, DEFAULT_LEARNING_RATE, DEFAULT_MIN_SEQUENCE_LENGTH
    from model import MidiNet, MidiNetConfig
    from checkpoint import load_checkpoint_train, save_checkpoint
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
        sequence = torch.tensor(self.music_sequences[index], dtype=int)
        return sequence[:-1], sequence[1:]


class MidiDatasetSampler(Sampler[list[int]]):
    """
    MidiDataset 的动态批次采样器。
    根据每个样本的序列长度动态分组，确保每个批次中 token 的总量不超过设定限制，从而实现高效训练。

    工作流程：
        1. 接收数据集和最大 token 限制；
        2. 返回一个自定义迭代器 MidiDatasetSamplerIter；
        3. 该迭代器会根据样本长度排序并动态打包成批次；
        4. 每个批次长度近似，避免 padding 浪费；
        5. 批次打乱以增加训练多样性。

    Args:
        dataset: 输入的数据集，必须包含 music_sequences 属性；
        max_batch_tokens: 每个批次中允许的最大 token 总数；
        sampler: 可选的外部采样器，用于控制索引顺序（如 DistributedSampler）。

    Returns:
        返回由样本索引列表构成的批次，供 DataLoader 使用。

    Examples:
        >>> dataset = MidiDataset("data/")
        >>> sampler = MidiDatasetSampler(dataset, max_batch_tokens=4096)
        >>> for batch in sampler:
        ...     print(batch)  # [19, 89, 64]
    """

    def __init__(self, dataset: MidiDataset, max_batch_tokens: int, sampler: Optional[Sampler[int]] = None):
        super().__init__()
        self.dataset = dataset
        self.max_batch_tokens = max_batch_tokens
        self.sampler = sampler
        self.iter = None

    def total_tokens(self) -> int:
        if not self.iter:
            iter(self)
        return self.iter.total_tokens

    def __iter__(self) -> "MidiDatasetSamplerIter":
        self.iter = MidiDatasetSamplerIter(self.dataset, self.max_batch_tokens, self.sampler)
        return self.iter

    def __len__(self) -> int:
        if not self.iter:
            iter(self)
        return len(self.iter)


class MidiDatasetSamplerIter(Iterator[list[int]]):
    """
    MidiDataset 的批次迭代器。
    按照样本序列长度对索引进行排序与分组，生成符合最大 token 限制的批次索引，并在每轮迭代时打乱批次顺序。

    工作流程：
        1. 获取输入索引（可自定义采样器）；
        2. 根据序列长度升序排序，长度相同时引入扰动打乱；
        3. 动态打包：每当当前批次无法容纳新样本时就结束该批；
        4. 最后将所有批次打乱；
        5. 支持 __len__ 返回真实批次数总 token 数。

    Args:
        dataset: 包含 music_sequences 属性的数据集；
        max_batch_tokens: 每个批次允许的最大 token 总数；
        sampler: 可选采样器，决定样本索引顺序。

    Returns:
        每次返回一个索引列表，表示一个批次的样本索引。
    """

    def __init__(self, dataset: MidiDataset, max_batch_tokens: int, sampler: Optional[Sampler[int]] = None):
        self.dataset = dataset
        self.max_batch_tokens = max_batch_tokens

        # 获取全部采样索引（如果有自定义采样器则使用）
        indices = list(sampler) if sampler else list(range(len(dataset)))

        # 计算每个样本的长度
        index_and_lengths = [(idx, len(dataset.music_sequences[idx])) for idx in indices]

        # 按长度排序，长度相同引入扰动避免排序固定
        sorted_pairs = sorted(index_and_lengths, key=lambda pair: (pair[1], random.random()))

        batches: list[list[int]] = []
        current_batch: list[int] = []
        self.total_tokens = 0

        for idx, seq_len in sorted_pairs:
            # 过长样本单独为一个批次
            if seq_len - 1 > max_batch_tokens:
                # 这里（以及下面）减去 1 是因为 inputs 的维度始终比原序列少 1（input = seq[:-1], lables = seq[1:]）
                self.total_tokens += seq_len - 1
                batches.append([idx])
                continue

            # 预测当前批次加上该样本后的 token 总数
            estimated_tokens = (len(current_batch) + 1) * (seq_len - 1)
            if estimated_tokens > max_batch_tokens:
                # 所有样本已按序列长度升序排序，因此 current_batch[-1] 一定是该批次中最长的
                self.total_tokens += len(current_batch) * (len(self.dataset.music_sequences[current_batch[-1]]) - 1)
                batches.append(current_batch)
                current_batch = []

            current_batch.append(idx)

        if current_batch:
            self.total_tokens += (len(self.dataset.music_sequences[current_batch[-1]]) - 1) * len(current_batch)
            batches.append(current_batch)

        # 打乱批次顺序，增加训练扰动
        random.shuffle(batches)

        # 存储批次列表迭代器和长度
        self._batch_iter = iter(batches)
        self._batch_count = len(batches)

    def __iter__(self):
        return self

    def __next__(self) -> list[int]:
        return next(self._batch_iter)

    def __len__(self) -> int:
        return self._batch_count


def sequence_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_token: int):
    "将多个样本合成统一长度的batch。"
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

    # 初始化损失列表、触发OOM的输入形状列表
    losses = []
    oom_shapes = []  # shape: [batch_size, seq_len]

    # 用于梯度缩放
    scaler = GradScaler()

    # 创建进度条，显示训练进度
    dataloader_iter = iter(dataloader)
    progress_bar = tqdm(total=dataloader.batch_sampler.total_tokens(), disable=not show_progress)
    for inputs, labels in dataloader_iter:
        inputs, labels = inputs.to(device), labels.to(device)
        progress_n = inputs.size(0) * inputs.size(1)  # 进度条更新的步数
        optimizer.zero_grad()  # 清空梯度

        try:
            # 使用半精度节省显存
            with autocast(device.type if device else "cpu", dtype=torch.float16):
                outputs = model(inputs, padding_mask=inputs == pad_token).view(-1, vocab_size)  # 前向传播并 reshape 成二维张量
                loss = F.cross_entropy(outputs, labels.view(-1), ignore_index=pad_token)  # 计算损失

            scaler.scale(loss).backward()  # 反向传播
            scaler.step(optimizer)  # 更新模型参数
            scaler.update()  # 调整缩放因子

            losses.append(loss.item())  # 累积训练损失
            progress_bar.set_postfix(loss=loss.item())  # 更新进度条
        except torch.OutOfMemoryError:
            # 记录OOM时的输入形状
            oom_shapes.append(list(inputs.shape))

            # 消除引用，方便垃圾回收
            inputs = labels = outputs = loss = None

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
    对模型进行验证，返回平均损失和平均困惑度。

    此函数遍历验证集，对模型进行前向传播，计算每个样本的交叉熵损失。
    返回所有样本的损失，以衡量模型在整个验证集上的性能表现。

    Args:
        model: 需要验证的 MidiNet 模型。
        dataloader: 验证数据加载器。
        vocab_size: 词表大小，用于 reshape 输出。
        pad_token: padding 的 token 值，用于掩码处理和损失忽略。
        device: 计算设备。
        show_progress: 是否显示进度条。

    Returns:
        验证损失。
    """
    # 清理缓存以释放内存
    empty_cache()

    # 初始化损失列表
    losses = []

    # 遍历整个验证集，不进行梯度计算
    dataloader_iter = iter(dataloader)
    progress_bar = tqdm(total=dataloader.batch_sampler.total_tokens(), disable=not show_progress)
    for inputs, labels in dataloader_iter:
        # 将输入移动到计算设备
        inputs, labels = inputs.to(device), labels.to(device)

        # 使用半精度节省显存
        with autocast(device.type if device else "cpu", dtype=torch.float16):
            # 模型前向传播，得到输出并 reshape 成二维张量
            outputs = model(inputs, padding_mask=inputs == pad_token).view(-1, vocab_size)

            # 计算并记录损失
            losses.extend(F.cross_entropy(outputs, labels.view(-1), ignore_index=pad_token, reduction="none").tolist())

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
          - train_loss(dict[str, Any]): 每个epoch的训练损失数量、平均值、标准差
          - val_loss(dict[str, Any]): 每个epoch的验证损失平均值、标准差
        img_path: 图形保存的文件路径，可以是字符串或Path对象。
    """
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 计算验证点的x坐标（每个epoch的起始位置）
    current_iteration = len(metrics["train_loss"][0]["count"])  # 当前累计的迭代次数
    val_iteration_points = [current_iteration]  # 存储每个epoch的起始迭代次数
    for epoch in metrics["train_loss"][1:]:
        current_iteration += epoch["count"]  # 累加当前epoch的迭代次数
        val_iteration_points.append(current_iteration)

    # 计算训练损失曲线的x坐标（偏移半个epoch）
    train_x = [val_iteration_points[i] - epoch["count"] / 2 for i, epoch in enumerate(metrics["train_loss"])]

    # 绘制训练损失曲线和标准差区间
    ax.plot(train_x, [epoch["mean"] for epoch in metrics["train_loss"]], label="Train Loss", color="red", linestyle="-", marker=".")
    ax.fill_between(train_x, *zip((epoch["mean"] + epoch["std_dev"], epoch["mean"] - epoch["std_dev"]) for epoch in metrics["train_loss"]), color="red", alpha=0.2)

    # 绘制验证损失曲线和标准差区间
    ax.plot(val_iteration_points, [epoch["mean"] for epoch in metrics["val_loss"]], label="Validation Loss", color="blue", linestyle="-", marker=".")
    ax.fill_between(val_iteration_points, *zip((epoch["mean"] + epoch["std_dev"], epoch["mean"] - epoch["std_dev"]) for epoch in metrics["val_loss"]), color="blue", alpha=0.2)

    # 设置X轴为整数刻度
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 设置坐标轴标签和标题
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    ax.set_title("Training Process")

    # 添加图例和网格
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 保存并展示图形
    plt.tight_layout()
    pathlib.Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    "解析命令行参数。"
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
    parser.add_argument("-b", "--train-max-batch-tokens", default=16384, type=int, help="训练时，每个批次的序列长度的和上限")
    parser.add_argument("-q", "--val-max-batch-tokens", default=32678, type=int, help="验证时，每个批次的序列长度的和上限")
    parser.add_argument("-l", "--learning-rate", default=DEFAULT_LEARNING_RATE, type=float, help="学习率")
    parser.add_argument("-w", "--weight-decay", default=DEFAULT_WEIGHT_DECAY, type=float, help="权重衰减系数")
    parser.add_argument("-n", "--num-heads", default=DEFAULT_NUM_HEADS, type=int, help="多头注意力中的注意力头数量")
    parser.add_argument("-d", "--dim-head", default=DEFAULT_DIM_HEAD, type=int, help="多头注意力中的注意力头的维度")
    parser.add_argument("-f", "--dim-feedforward", default=DEFAULT_DIM_FEEDFORWARD, type=int, help="前馈神经网络的隐藏层维度")
    parser.add_argument("-s", "--num-layers", default=DEFAULT_NUM_LAYERS, type=int, help="模型 Transformer 编码器中的层数")
    parser.add_argument("-o", "--dropout", default=DEFAULT_DROPOUT, type=float, help="Dropout 概率，用于防止过拟合")

    # 解析命令行参数并返回
    return parser.parse_args()


def _mp_fn(rank: int, world_size: int, args: argparse.Namespace):
    # 部署分布式训练环境
    if world_size > 1:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # 获取设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # 清理缓存以释放内存
    with device:
        empty_cache()

    # 加载检查点
    tokenizer, model_state_dict, optimizer_state_dict, metrics = load_checkpoint_train(args.ckpt_path)

    # 加载训练集并创建数据加载器
    train_dataset = MidiDataset(args.train_dataset, tokenizer, args.min_sequence_length, args.max_sequence_length, show_progress=rank == 0)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_sampler=MidiDatasetSampler(train_dataset, args.train_max_batch_tokens, train_sampler), collate_fn=lambda x: sequence_collate_fn(x, pad_token=tokenizer.pad_token_id))

    # 如果有的话，加载验证集并创建数据加载器
    if args.val_dataset:
        val_dataset = MidiDataset(args.val_dataset, tokenizer, args.min_sequence_length, args.max_sequence_length, show_progress=rank == 0)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(val_dataset, batch_sampler=MidiDatasetSampler(val_dataset, args.val_max_batch_tokens, val_sampler), collate_fn=lambda x: sequence_collate_fn(x, pad_token=tokenizer.pad_token_id))

    # 初始化模型
    model = MidiNet(MidiNetConfig(len(tokenizer), args.num_heads, args.dim_head, args.dim_feedforward, args.num_layers), dropout=args.dropout)

    # 加载模型状态
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError:
        metrics = {"train_loss": [], "val_loss": []}

    # 转移模型到设备
    model = model.to(device)

    # 用 DDP 包装模型
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model)

    # 创建优化器
    optimizer = optim.AdamW(model.parameters())

    # 加载优化器状态
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    # 设置学习率、权重衰减系数
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
        group["weight_decay"] = args.weight_decay

    # 开始训练模型
    for epoch in range(args.num_epochs):
        # 训练一轮模型
        train_sampler.set_epoch(epoch)
        train_loss, oom_shapes = train(model, train_loader, optimizer, len(tokenizer), tokenizer.pad_token_id, device, show_progress=rank == 0)

        # 如果指定了验证集，就进行验证，否则跳过验证并设置验证损失为 NaN
        if args.val_dataset:
            val_sampler.set_epoch(epoch)
            val_loss = validate(model, val_loader, len(tokenizer), tokenizer.pad_token_id, device, show_progress=rank == 0)
        else:
            val_loss = [float("nan")]

        # 将所有进程的损失汇集
        train_loss = np.array(itertools.chain(dist.gather_object(train_loss)))
        val_loss = np.array(itertools.chain(dist.gather_object(val_loss)))

        # 计算并添加损失平均值和标准差到指标
        metrics["train_loss"].append({"mean": train_loss.mean(), "std_dev": train_loss.std(), "count": len(train_loss)})
        metrics["val_loss"].append({"mean": val_loss.mean(), "std_dev": val_loss.std()})

    # 将所有进程中使内存爆炸的张量的形状汇集
    oom_shapes = list(itertools.chain(dist.gather_object(oom_shapes)))

    if rank == 0:
        # 保存最后一次训练时使内存爆炸的张量的形状
        if oom_shapes:
            with open("oom_shapes.txt", "w", encoding="utf-8") as f:
                f.write("Shape (e.g: Batch Size x Sequence Length)\n")
                f.write("\n".join(f"{batch_size} x {sequence_length}" for batch_size, sequence_length in oom_shapes))

        # 保存当前模型的检查点
        save_checkpoint(model, optimizer.state_dict(), metrics, args.ckpt_path)

        # 绘制训练过程中的损失曲线
        plot_training_process(metrics, "statistics.png")

    # 释放资源
    if world_size > 1:
        dist.destroy_process_group()


def main():
    # 解析命令行参数
    args = parse_args()

    # 如果有多 GPU，使用 DDP 加速训练
    world_size = torch.cuda.device_count()
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        mp.spawn(_mp_fn, (world_size, args), nprocs=world_size)
    else:
        _mp_fn(0, 1, args)


if __name__ == "__main__":
    main()
