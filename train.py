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
from torch import nn, optim
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import PreTrainedTokenizerFast
from collections.abc import Callable

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
        self.length = 0

        # 计算每个批次的大小
        self.batches_size = [0]
        cur_batch_size = 0

        # 按序列长度排序
        for sequence in sorted(self.dataset.music_sequences, key=lambda x: len(x)):
            # 计算序列长度平方
            sequence_size = len(sequence) ** 2

            # 跳过长度平方大于 max_batch_size 的序列
            if sequence_size > self.max_batch_size:
                continue

            # 如果当前批次的长度平方和大于 max_batch_size，则保存并清空当前批次
            if cur_batch_size + sequence_size > self.max_batch_size:
                self.batches_size.append(0)
                cur_batch_size = 0

            # 添加样本，累积当前批次大小
            self.batches_size[-1] += 1
            cur_batch_size += sequence_size

        # 如果没有序列剩余或要求丢弃最后一个批次
        if self.drop_last or not self.batches_size[-1]:
            self.batches_size.pop(-1)

    def __iter__(self):
        # 获取数据集的长度
        length = len(self.dataset)

        # 初始化批次列表
        batches = []
        batch = []

        # 先按序列长度排序，如果序列长度相同就随机排序
        for index, sequence in sorted(zip(range(length), self.dataset.music_sequences), key=lambda x: len(x[1])):
            # 跳过长度平方大于 max_batch_size 的序列
            if len(sequence) ** 2 > self.max_batch_size:
                continue

            # 添加样本
            batch.append(index)

            # 如果达到批次大小，则下一个批次
            if len(batch) == self.batches_size[len(batches)]:
                batches.append(batch)
                batch = []

        # 打乱所有批次顺序
        random.shuffle(batches)

        # 返回每个批次
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.batches_size)


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
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    vocab_size: int,
    pad_token: int = 0,
    device: torch.device = None,
    pbar_desc: str = "训练"
) -> tuple[list[float], float]:
    """
    训练模型的函数。
    此函数进行一轮训练，逐步优化模型参数，输出每一步的训练损失和平均困惑度。

    工作流程:
        1. 初始化数据加载器。
        2. 将模型移动到指定设备。
        3. 选择交叉熵损失函数。
        4. 使用进度条显示训练进度。
        5. 切换模型到训练模式。
        6. 对每个batch进行前向传播、计算损失、反向传播和更新参数。
        7. 累积损失和困惑度。
        8. 返回这个epoch每一步的训练损失和困惑度平均值。

    Args:
        model: 需要训练的神经网络模型。
        dataloader: 训练数据加载器。
        optimizer: 用于优化模型的优化器。
        criterion: 指定的损失函数。通常使用交叉熵损失函数。
        vocab_size: 词汇表的大小，用于调整输出层的维度。
        pad_token: 填充token的标记，用于忽略计算损失。
        device: 指定训练的设备。

    Returns:
        该epoch每一步的训练损失和训练困惑度的平均值。

    Examples:
        >>> tokenizer = PreTrainedTokenizerFast.from_pretrained("ckpt/tokenizer")
        >>> model = MidiNet(len(tokenizer), 768, 12, 2048, 12)
        >>> optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
        >>> criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        >>> train(model, dataset, optimizer, criterion, len(tokenizer), tokenizer.pad_token_id)
        ([1.9, 0.89, 0.6, 0.4], 42.6)
    """
    # 清理缓存以释放内存
    empty_cache()

    # 将模型移动到设备
    model = model.to(device)

    # 设置模型为训练模式
    model.train()
    train_loss, train_ppl = [], []  # 初始化损失、困惑度列表

    # 创建进度条，显示训练进度
    for inputs, labels in tqdm(dataloader, desc=pbar_desc):
        inputs, labels = inputs.to(device), labels.to(device).view(-1)
        optimizer.zero_grad()  # 清空梯度
        try:
            outputs = model(inputs, padding_mask=inputs == pad_token).view(-1, vocab_size)  # 前向传播并 reshape 成二维张量
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
        except torch.cuda.OutOfMemoryError:
            empty_cache()
            continue

        train_ppl.append(torch.exp(loss).item())  # 累积训练困惑度
        train_loss.append(loss.item())  # 累积训练损失

    return train_loss, sum(train_ppl) / len(train_ppl)


def validate(
    model: MidiNet,
    dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    vocab_size: int,
    pad_token: int = 0,
    device: torch.device = None,
    pbar_desc: str = "验证"
) -> tuple[float, float]:
    """
    对模型进行验证，返回平均损失和平均困惑度。

    此函数遍历验证集，对模型进行前向传播，计算每个 batch 的交叉熵损失和困惑度。
    所有 batch 的损失与困惑度将被平均后返回，以衡量模型在整个验证集上的性能表现。

    Args:
        model: 需要验证的 MidiNet 模型。
        dataloader: 验证数据加载器。
        criterion: 指定的损失函数。通常使用交叉熵损失函数。
        vocab_size: 词表大小，用于 reshape 输出。
        pad_token: padding 的 token 值，用于掩码处理和损失忽略。
        device: 计算设备。

    Returns:
        平均损失和平均困惑度。
    """
    # 清理缓存以释放内存
    empty_cache()

    # 初始化损失、困惑度列表
    val_loss = []
    val_ppl = []

    # 遍历整个验证集，不进行梯度计算
    for inputs, labels in tqdm(dataloader, desc=pbar_desc):
        with torch.no_grad():
            # 将输入移动到计算设备
            inputs, labels = inputs.to(device), labels.to(device).view(-1)

            # 模型前向传播，得到输出并 reshape 成二维张量
            outputs = model(inputs, padding_mask=inputs == pad_token).view(-1, vocab_size)

            # 计算损失
            loss = criterion(outputs, labels)

            # 记录损失和困惑度
            val_loss.append(loss.item())
            val_ppl.append(torch.exp(loss).item())

    # 返回平均损失和平均困惑度
    return sum(val_loss) / len(val_loss), sum(val_ppl) / len(val_ppl)


def plot_training_process(metries: dict[str, list], img_path: pathlib.Path | str):
    """
    绘制训练过程中的损失、困惑度曲线。

    Args:
        metries: 指标，包含
          - train_loss(list[list[float]]): 每个epoch的每一步的训练损失
          - train_ppl(list[float]): 每个epoch的训练困惑度平均值
          - val_loss(list[float]): 每个epoch的验证损失平均值
          - val_ppl(list[float]): 每个epoch的验证困惑度平均值
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

    # 创建第二个Y轴用于困惑度
    ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴

    # 绘制训练过程中的困惑度曲线
    ax2.plot([x - 0.5 for x in val_steps], metries["train_ppl"], label="Train Perplexity", marker=".", color="green", linestyle="--")

    # 绘制验证过程中的困惑度曲线
    ax2.plot(val_steps, metries["val_ppl"], label="Validation Perplexity", marker=".", color="blue", linestyle="--")

    # 设置第二个Y轴的标签
    ax2.set_ylabel("Perplexity")

    # 设置标题和图例
    ax1.set_title("Training Process")  # 设置标题
    ax1.legend(loc="upper left")  # 为损失曲线添加图例
    ax2.legend(loc="upper right")  # 为困惑度曲线添加图例

    plt.tight_layout()  # 自动调整布局，防止标签重叠
    pathlib.Path(img_path).parent.mkdir(parents=True, exist_ok=True)  # 确保保存路径存在
    plt.savefig(img_path, dpi=300, bbox_inches="tight")  # 保存图形
    plt.show()  # 显示图形


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="训练 MIDI 模型并绘制训练过程中的损失、困惑度曲线。")
    parser.add_argument("num_epochs", type=int, help="训练的总轮数")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="加载和保存检查点的路径")
    parser.add_argument("-t", "--train-dataset", action="append", type=pathlib.Path, required=True, help="训练集文件路径（可多次指定以使用多个数据集）")
    parser.add_argument("-v", "--val-dataset", action="append", type=pathlib.Path, help="验证集文件路径（可多次指定以使用多个数据集）")
    parser.add_argument("-m", "--min-sequence-length", default=DEFAULT_MIN_SEQUENCE_LENGTH, type=int, help="最小序列长度，小于该长度的样本不会被训练")
    parser.add_argument("-e", "--max-sequence-length", default=2 ** 17, type=int, help="最大序列长度，大于该长度的样本将被截断")
    parser.add_argument("-b", "--max-batch-size", default=8 * 1536 ** 2, type=int, help="每个批次的序列长度的平方和上限")
    parser.add_argument("-l", "--learning-rate", default=DEFAULT_LEARNING_RATE, type=float, help="学习率")
    parser.add_argument("-w", "--weight-decay", default=DEFAULT_WEIGHT_DECAY, type=float, help="权重衰减系数")
    parser.add_argument("-n", "--num-heads", default=DEFAULT_NUM_HEADS, type=int, help="多头注意力中的注意力头数量")
    parser.add_argument("-d", "--dim-head", default=DEFAULT_DIM_HEAD, type=int, help="多头注意力中的注意力头的维度")
    parser.add_argument("-f", "--dim-feedforward", default=DEFAULT_DIM_FEEDFORWARD, type=int, help="前馈神经网络的隐藏层维度")
    parser.add_argument("-s", "--num-layers", default=DEFAULT_NUM_LAYERS, type=int, help="模型 Transformer 编码器中的层数")
    parser.add_argument("-o", "--dropout", default=DEFAULT_DROPOUT, type=float, help="Dropout 概率，用于防止过拟合")
    args = parser.parse_args()

    # 清理缓存以释放内存
    empty_cache()

    # 加载检查点
    tokenizer, model_state, optimizer_state, metries = load_checkpoint(args.ckpt_path, train=True)

    # 加载训练集并创建数据加载器
    train_dataset = MidiDataset(args.train_dataset, tokenizer, args.min_sequence_length, args.max_sequence_length)
    train_loader = DataLoader(train_dataset, batch_sampler=MidiDatasetSampler(train_dataset, args.max_batch_size), collate_fn=lambda x: sequence_collate_fn(x, pad_token=tokenizer.pad_token_id))

    # 如果有的话，加载验证集并创建数据加载器
    if args.val_dataset:
        val_dataset = MidiDataset(args.val_dataset, tokenizer, args.min_sequence_length, args.max_sequence_length)
        val_loader = DataLoader(val_dataset, batch_sampler=MidiDatasetSampler(val_dataset, args.max_batch_size), collate_fn=lambda x: sequence_collate_fn(x, pad_token=tokenizer.pad_token_id))

    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = MidiNet(len(tokenizer), args.num_heads, args.dim_head, args.dim_feedforward, args.num_layers, dropout=args.dropout)

    # 加载模型状态
    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        metries = {"val_ppl": [], "train_ppl": [], "val_loss": [], "train_loss": []}

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

    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 开始训练模型
    for epoch in range(args.num_epochs):
        # 训练并添加损失和困惑度到指标
        train_loss, train_ppl = train(model, train_loader, optimizer, criterion, len(tokenizer), tokenizer.pad_token_id, device, f"训练第 {epoch + 1} 个 Epoch")
        metries["train_loss"].append(train_loss)
        metries["train_ppl"].append(train_ppl)

        # 如果指定了验证集，就进行验证，否则跳过验证并设置验证损失和困惑度为 NaN
        if args.val_dataset:
            val_loss, val_ppl = validate(model, val_loader, criterion, len(tokenizer), tokenizer.pad_token_id, device, f"验证第 {epoch + 1} 个 Epoch")
        else:
            val_loss = val_ppl = float("nan")

        # 添加验证损失和困惑度到指标
        metries["val_loss"].append(val_loss)
        metries["val_ppl"].append(val_ppl)

    # 保存当前模型的检查点
    save_checkpoint(model, optimizer, metries, args.ckpt_path)

    # 绘制训练过程中的损失和困惑度曲线
    plot_training_process(metries, "statistics.png")


if __name__ == "__main__":
    main()
