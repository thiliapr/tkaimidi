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
import orjson
import os
from multiprocessing import cpu_count
from typing import Optional, Iterator
import mido
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import PreTrainedTokenizerFast
from constants import DEFAULT_DIM_HEAD, DEFAULT_NUM_HEADS, DEFAULT_DIM_FEEDFORWARD, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT, DEFAULT_WEIGHT_DECAY, DEFAULT_LEARNING_RATE, DEFAULT_MIN_SEQUENCE_LENGTH
from model import MidiNet, MidiNetConfig
from checkpoint import load_checkpoint_train, save_checkpoint
from utils import midi_to_notes, notes_to_sheet, empty_cache, parallel_map
from tokenizer import data_to_str

# 解除线程数量限制
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count())
torch.set_num_threads(cpu_count())


class MidiDataset(Dataset):
    """
    MIDI 数据集类，用于加载、处理 MIDI/JSON 文件并转换为模型训练所需的分词序列。

    工作流程:
        1. 遍历指定目录下所有 MIDI 和 JSON 文件。
        2. 对 MIDI 文件，读取后转为音符序列，再转为乐谱分词序列。
        3. 对 JSON 文件，直接读取分词序列。
        4. 跳过过短的序列，截断过长的序列。
        5. 支持多进程并行处理文件，加速数据加载。
        6. 所有序列存储于 self.music_sequences，支持索引访问和长度查询。

    Attributes:
        music_sequences: 存储所有训练样本的分词序列列表。

    Args:
        midi_dirs: 包含 MIDI/JSON 文件的目录列表。
        tokenizer: 用于音乐数据编码的分词器。
        min_sequence_length: 训练序列的最小长度，短于该长度的样本会被丢弃。（分词后的序列长度，包括开始和结束标志）
        max_sequence_length: 训练序列的最大长度，长于该长度的样本会被截断。（分词后的序列长度，包括开始和结束标志）
        seed: 随机种子，用于数据加载时的随机性控制。

    Examples:
        >>> dataset = MidiDataset(
        ...     midi_dirs=[pathlib.Path("data/midi")],
        ...     tokenizer=tokenizer,
        ...     min_sequence_length=64,
        ...     max_sequence_length=8964
        ... )
        >>> len(dataset)
        198964
        >>> dataset[0]
    """

    def __init__(
        self,
        midi_dirs: list[pathlib.Path],
        tokenizer: PreTrainedTokenizerFast,
        min_sequence_length: int,
        max_sequence_length: int,
        seed: int = 8964
    ):
        self.music_sequences = []  # 存储所有训练样本的分词序列
        num_workers = cpu_count()  # 并行处理的进程数

        # 收集所有 MIDI 文件路径
        midi_files = [f for dir_path in midi_dirs for f in dir_path.rglob("*.*") if f.is_file() and f.suffix.lower() in {".mid", ".midi", }]
        random.Random(seed).shuffle(midi_files)

        # 并行处理 MIDI 文件
        midi_chunks = [midi_files[i::num_workers] for i in range(num_workers)]
        midi_results = parallel_map(self.process_midi_files, [(chunk, max_sequence_length, min_sequence_length, tokenizer) for chunk in midi_chunks], num_workers=num_workers)

        # 扁平化结果
        self.music_sequences.extend([seq for sublist in midi_results for seq in sublist])

        # 收集所有 JSON 文件路径
        json_files = [f for dir_path in midi_dirs for f in dir_path.rglob("*.json") if f.is_file() and f.suffix.lower() == ".json"]
        random.Random(seed).shuffle(json_files)

        # 并行处理 JSON 文件
        json_chunks = [json_files[i::num_workers] for i in range(num_workers)]
        json_results = parallel_map(self.process_json_files, [(chunk, max_sequence_length, min_sequence_length, tokenizer) for chunk in json_chunks], num_workers=num_workers)

        # 扁平化结果
        self.music_sequences.extend([seq for sublist in json_results for seq in sublist])

    @staticmethod
    def process_midi_files(files: list[pathlib.Path], max_sequence_length: int, min_sequence_length: int, tokenizer: PreTrainedTokenizerFast) -> list[list[int]]:
        "并行处理 MIDI 文件，将其转为分词序列。"
        result = []
        for file_path in files:
            try:
                midi_file = mido.MidiFile(file_path, clip=True)
            except (ValueError, EOFError, OSError):
                # 跳过损坏或无法读取的 MIDI 文件
                continue

            notes = midi_to_notes(midi_file)
            sheet, _ = notes_to_sheet(notes, max_length=max_sequence_length)

            # 截断超长序列
            seq = tokenizer.encode(data_to_str(sheet), max_length=max_sequence_length, truncation=True)

            # 如果分词序列长度小于最小长度，则跳过
            if len(seq) < min_sequence_length:
                continue

            # 编码为分词序列
            result.append(seq)
        return result

    @staticmethod
    def process_json_files(files: list[pathlib.Path], max_sequence_length: int, min_sequence_length: int, tokenizer: PreTrainedTokenizerFast) -> list[list[int]]:
        "并行处理 JSON 文件，直接读取分词序列。"
        result = []
        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = orjson.loads(f.read())
            except orjson.JSONDecodeError:
                continue

            # 截断超长序列
            seq = tokenizer.encode(data["data"], max_length=max_sequence_length, truncation=True)

            # 如果分词序列长度小于最小长度，则跳过
            if len(seq) < min_sequence_length:
                continue

            result.append(seq)
        return result

    def __len__(self) -> int:
        return len(self.music_sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.tensor(self.music_sequences[index], dtype=torch.long)
        return sequence[:-1], sequence[1:]


class MidiDatasetSampler(Sampler[list[int]]):
    """
    用于 MIDI 数据集的分批采样器，根据序列长度进行动态批处理。

    该采样器会:
    1. 根据序列长度对样本进行排序
    2. 动态创建批次，确保每个批次的token总数不超过max_batch_tokens
    3. 每个epoch都会重新打乱数据顺序

    Attributes:
        max_batch_tokens: 单个批次允许的最大token数量
        seed: 随机种子
        batches: 当前分配到的批次列表
        total_tokens: 当前分配到的总token数

    Examples:
        >>> dataset = MidiDataset([pathlib.Path("data/")], tokenizer, min_sequence_length=64, max_sequence_length=8964)
        >>> sampler = MidiDatasetSampler(dataset, max_batch_tokens=4096)
        >>> for batch in sampler:
        ...     print(batch)  # [19, 89, 64]
    """

    def __init__(self, dataset: MidiDataset, max_batch_tokens: int, seed: int = 0):
        super().__init__()
        self.max_batch_tokens = max_batch_tokens
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

        # 批次倒序，用于快速检测训练的问题
        batches_with_tokens.reverse()

        # 分配批次
        self.batches = [batch for batch, _ in batches_with_tokens]
        self.total_tokens = sum(tokens for _, tokens in batches_with_tokens)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def sequence_collate_fn(batch, pad_token: int = 0):
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
    progress_bar = tqdm(total=dataloader.batch_sampler.total_tokens)

    # 初始化损失列表、触发OOM的输入形状列表
    losses = []
    oom_shapes = []  # shape: [batch_size, seq_len]

    # 提前确定 device_type，避免多次判断
    device_type = device.type if device is not None else "cpu"

    # 遍历整个训练集
    for inputs, labels in dataloader_iter:
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
    device: Optional[torch.device] = None
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
    progress_bar = tqdm(total=dataloader.batch_sampler.total_tokens)

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
                {"mean": 1.2, "std": 0.1, "count": 100},
                {"mean": 1.0, "std": 0.08, "count": 100},
            ],
            "val_loss": [
                {"mean": 1.1, "std": 0.09},
                {"mean": 0.95, "std": 0.07},
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
    train_upper = [epoch["mean"] + epoch["std"] for epoch in metrics["train_loss"]]
    train_lower = [epoch["mean"] - epoch["std"] for epoch in metrics["train_loss"]]
    ax.fill_between(train_x, train_upper, train_lower, color="red", alpha=0.2)

    ax.plot(val_iteration_points, [epoch["mean"] for epoch in metrics["val_loss"]], label="Validation Loss", color="blue", linestyle="-", marker=".")
    val_upper = [epoch["mean"] + epoch["std"] for epoch in metrics["val_loss"]]
    val_lower = [epoch["mean"] - epoch["std"] for epoch in metrics["val_loss"]]
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
    "解析训练 MIDI 模型的命令行参数。"
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="训练 MIDI 模型并绘制训练过程中的损失、困惑度曲线。")

    # 添加必须参数
    parser.add_argument("num_epochs", type=int, help="训练的总轮数")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="加载和保存检查点的路径")
    parser.add_argument("-t", "--train-dataset", action="append", type=pathlib.Path, required=True, help="训练集文件路径（可多次指定以使用多个数据集）")

    # 添加可选参数
    parser.add_argument("-v", "--val-dataset", action="append", type=pathlib.Path, help="验证集文件路径（可多次指定以使用多个数据集）")
    parser.add_argument("-m", "--min-sequence-length", default=DEFAULT_MIN_SEQUENCE_LENGTH, type=int, help="训练时，分词序列的最小长度，短于该长度的样本会被丢弃，默认为 %(default)s")
    parser.add_argument("-e", "--max-sequence-length", default=2048, type=int, help="训练时，分词序列的最大长度，长于该长度的样本会被截断，默认为 %(default)s")
    parser.add_argument("-b", "--train-max-batch-tokens", default=8192, type=int, help="训练时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-q", "--val-max-batch-tokens", default=16384, type=int, help="验证时，每个批次的序列长度的和上限，默认为 %(default)s")
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


def main(args: argparse.Namespace):
    """
    主训练入口函数。

    Args:
        args: 包含训练所需参数的命名空间对象，包括数据集路径、模型参数、训练超参数等。

    功能:
        1. 设置设备（GPU/CPU）和随机种子，确保训练的可复现性。
        2. 加载训练检查点，包括分词器、模型权重、优化器状态和历史指标。
        3. 构建训练和验证数据集及其采样器和数据加载器。
        4. 初始化模型结构，并加载权重到指定设备。
        5. 配置优化器及其参数（学习率、权重衰减等）。
        6. 进行多轮训练，每轮包括训练和（可选的）验证，记录损失和相关指标。
        7. 记录并保存训练过程中出现 OOM（内存溢出）时的张量形状信息。
        8. 保存最终模型检查点和训练指标。
        9. 绘制并保存训练过程中的损失曲线图。
    """
    # 设置当前进程的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 清理缓存以释放内存
    empty_cache()

    # 加载训练检查点（包括 tokenizer、模型、优化器状态、指标）
    tokenizer, model_state_dict, optimizer_state_dict, metrics = load_checkpoint_train(args.ckpt_path)

    # 加载训练数据集及分布式采样器
    train_dataset = MidiDataset(args.train_dataset, tokenizer, args.min_sequence_length, args.max_sequence_length)
    train_sampler = MidiDatasetSampler(train_dataset, args.train_max_batch_tokens, args.seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=lambda x: sequence_collate_fn(x, pad_token=tokenizer.pad_token_id))

    # 如果存在验证集，加载验证数据集
    if args.val_dataset:
        val_dataset = MidiDataset(args.val_dataset, tokenizer, args.min_sequence_length, args.max_sequence_length)
        val_sampler = MidiDatasetSampler(val_dataset, args.val_max_batch_tokens, args.seed)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=lambda x: sequence_collate_fn(x, pad_token=tokenizer.pad_token_id))

    # 设置随机种子，确保可复现性
    set_seed(args.seed)

    # 初始化模型结构
    model = MidiNet(MidiNetConfig(len(tokenizer), args.num_heads, args.dim_head, args.dim_feedforward, args.num_layers), dropout=args.dropout)

    # 加载模型权重
    if model_state_dict:
        model.load_state_dict(model_state_dict)

    # 转移模型到设备
    model = model.to(device)

    # 多 GPU 时使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

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
        set_seed(args.seed + current_epoch)

        # 训练一轮模型
        train_sampler.set_epoch(current_epoch)
        train_loss, oom_shapes = train(model, train_loader, optimizer, len(tokenizer), tokenizer.pad_token_id, device)

        # 如果指定了验证集，就进行验证，否则跳过验证并设置验证损失为 NaN
        if args.val_dataset:
            val_sampler.set_epoch(current_epoch)
            val_loss = validate(model, val_loader, len(tokenizer), tokenizer.pad_token_id, device)
        else:
            val_loss = [float("nan")]

        # 计算并添加损失平均值和标准差到指标
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        metrics["train_loss"].append({"mean": train_loss.mean().item(), "std": train_loss.std().item(), "count": len(train_loss)})
        metrics["val_loss"].append({"mean": val_loss.mean().item(), "std": val_loss.std().item()})

    # 保存最后一次训练时使内存爆炸的张量的形状
    if oom_shapes:
        with open(args.ckpt_path / "oom_shapes.txt", "w", encoding="utf-8") as f:
            f.write("Shape (e.g: Batch Size x Sequence Length)\n")
            f.write("\n".join(f"{batch_size} x {sequence_length}" for batch_size, sequence_length in oom_shapes))

    # 保存当前模型的检查点
    save_checkpoint((model.module if torch.cuda.device_count() > 1 else model).cpu().state_dict(), optimizer.state_dict(), metrics, args.ckpt_path)
    print(f"训练完成，模型已保存到 {args.ckpt_path}。")

    # 绘制训练过程中的损失曲线
    plot_training_process(metrics, args.ckpt_path / "statistics.png")


if __name__ == "__main__":
    main(parse_args())
