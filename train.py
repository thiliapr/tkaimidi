# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import random
import pathlib
import argparse
from typing import Optional
from collections.abc import Callable, Iterator
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from generate import plot_piano_roll
from utils.checkpoint import load_checkpoint_train, save_checkpoint
from utils.constants import DEFAULT_ACCUMULATION_STEPS, DEFAULT_DECODER_DROPOUT, DEFAULT_ENCODER_DROPOUT, DEFAULT_LEARNING_RATE, DEFAULT_PIANO_ROLL_LENGTH, DEFAULT_PITCH_DROPOUT, DEFAULT_VARIANCE_PREDICTOR_DROPOUT, DEFAULT_WEIGHT_DECAY
from utils.model import MidiNet
from utils.toolkit import convert_to_tensor, create_padding_mask

# 解除线程数量限制
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())


class MidiDataset(Dataset):
    """
    MIDI 数据集加载器

    Args:
        dataset_file: 快速训练数据集文件

    Yields:
        - 钢琴卷帘
        - 音符数量（密度）
        - 平均音高
        - 音高范围（极差）
    """

    def __init__(self, dataset_file: os.PathLike):
        # 获取所有音频特征
        self.data_samples = np.load(dataset_file)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate([np.zeros([1, 128], dtype=bool), self.data_samples[f"{index}:piano_roll"]], axis=0),  # 添加全零帧作为起始帧
            self.data_samples[f"{index}:note_counts"],
            self.data_samples[f"{index}:pitch_means"],
            self.data_samples[f"{index}:pitch_ranges"]
        )

    def __len__(self) -> int:
        # 每个样本有 4 个特征，然后这个文件还储存了每个序列的长度，也就是 1 个特征
        # 所以用特征总数减去 1 再除以 4 就是样本数
        return (len(self.data_samples) - 1) // 4


class MidiDatasetSampler(Sampler[list[int]]):
    """
    用于 MIDI 数据集的分批采样器，根据序列长度进行动态批处理。

    该采样器会:
    1. 根据序列长度对样本进行排序
    2. 动态创建批次，确保每个批次的 token 总数不超过 max_batch_tokens
    3. 每个 epoch 都会重新打乱数据顺序

    Attributes:
        max_batch_tokens: 单个批次允许的最大 token 数量
        seed: 随机种子
        batches: 当前分配到的批次列表

    Examples:
        >>> dataset = MidiDataset("dataset.npz")
        >>> sampler = MidiDatasetSampler(dataset, max_batch_tokens=4096)
        >>> for batch in sampler:
        ...     print(batch)  # [19, 89, 64]
    """

    def __init__(self, dataset: MidiDataset, max_batch_tokens: int, seed: int = 8964):
        super().__init__()
        self.max_batch_tokens = max_batch_tokens
        self.seed = seed
        self.batches: list[list[int]]

        # 预计算所有样本的索引和长度
        length = dataset.data_samples["length"].tolist()
        self.index_and_lengths = [
            (idx, length[idx])
            for idx in range(len(dataset))
        ]
        self.index_to_length = dict(self.index_and_lengths)

    def set_epoch(self, epoch: int) -> None:
        """
        设置当前 epoch 并重新生成批次

        每个 epoch 开始时调用，用于:
        1. 根据新 epoch 重新打乱数据顺序
        2. 重新分配批次
        """
        generator = random.Random(self.seed + epoch)

        # 按长度排序，加入随机因子避免固定排序
        sorted_pairs = sorted(self.index_and_lengths, key=lambda pair: (pair[1], generator.random()))

        # 初始化批次列表
        self.batches = []
        current_batch = []

        # 遍历每一个样本
        for idx, seq_len in sorted_pairs:
            # 处理超长序列
            if seq_len > self.max_batch_tokens:
                self.batches.append([idx])
                continue

            # 计算当前批次加入新样本后的 token 总数
            estimated_tokens = (len(current_batch) + 1) * seq_len
            if estimated_tokens > self.max_batch_tokens:
                self.batches.append(current_batch)
                current_batch = []

            current_batch.append(idx)

        # 添加最后一个批次
        if current_batch:
            self.batches.append(current_batch)

        # 扰乱批次顺序，增强模型泛化能力
        generator.shuffle(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def sequence_collate_fn(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> tuple[
    torch.BoolTensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.BoolTensor,
]:
    """
    处理变长序列数据的批次整理函数
    将输入的多个变长序列样本整理为批次张量，包括钢琴卷帘、音符数量、平均音高、音高范围，并生成相应的填充掩码和序列长度信息

    工作流程：
    1. 解压批次数据并将每个特征转换为 PyTorch 张量
    2. 为钢琴卷帘创建填充掩码
    3. 计算序列的实际长度
    4. 对所有序列进行填充对齐处理
    5. 返回整理后的批次数据

    Args:
        batch: 包含多个样本的列表，每个样本是包含钢琴卷帘、音符数量、平均音高、音高范围特征的元组

    Returns:
        包含整理后批次数据的元组，包括填充后的钢琴卷帘、音符数量、平均音高、音高范围，以及填充掩码

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> dataset = MidiDataset("dataset.npz")
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=sequence_collate_fn)
        >>> for piano_roll, note_counts, pitch_means, pitch_ranges, padding_mask in dataloader:
        >>>     ...
    """
    # 解压批次数据并将每个特征列表转换为张量列表
    piano_roll, note_counts, pitch_means, pitch_ranges = [convert_to_tensor(item) for item in zip(*batch)]

    # 创建填充掩码用于标识有效数据位置
    padding_mask = create_padding_mask(piano_roll)

    # 对变长序列进行填充对齐，使批次内所有样本长度一致
    piano_roll, note_counts, pitch_means, pitch_ranges = [torch.nn.utils.rnn.pad_sequence(item, batch_first=True) for item in [piano_roll, note_counts, pitch_means, pitch_ranges]]

    # 返回批次数据
    return (
        piano_roll,
        note_counts,
        pitch_means,
        pitch_ranges,
        padding_mask,
    )


def midinet_loss(
    piano_roll_pred: torch.Tensor,
    note_counts_pred: torch.Tensor,
    pitch_mean_pred: torch.Tensor,
    pitch_range_pred: torch.Tensor,
    piano_roll_target: torch.Tensor,
    note_counts_target: torch.Tensor,
    pitch_mean_target: torch.Tensor,
    pitch_range_target: torch.Tensor,
    padding_mask: torch.BoolTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 MidiNet 模型的复合损失函数，包含钢琴卷帘、音符数量、音高均值和音高范围四个损失分量
    该函数通过掩码机制处理变长序列，仅对有效帧计算损失，忽略填充区域
    使用二元交叉熵计算钢琴卷帘损失，使用均方误差计算其他三个回归损失分量

    Args:
        piano_roll_pred: 预测的钢琴卷帘矩阵，形状为 [batch_size, seq_len, 128]
        note_counts_pred: 预测的音符数量，形状为 [batch_size, seq_len]
        pitch_mean_pred: 预测的平均音高，形状为 [batch_size, seq_len]
        pitch_range_pred: 预测的音高范围，形状为 [batch_size, seq_len]
        piano_roll_target: 目标的钢琴卷帘矩阵，形状为 [batch_size, seq_len, 128]
        note_counts_target: 目标的音符数量，形状为 [batch_size, seq_len]
        pitch_mean_target: 目标的平均音高，形状为 [batch_size, seq_len]
        pitch_range_target: 目标的音高范围，形状为 [batch_size, seq_len]
        padding_mask: 填充掩码，True 表示填充位置，形状为 [batch_size, seq_len]

    Returns:
        返回四个损失张量的元组：(钢琴卷帘损失, 音符数量损失, 音高均值损失, 音高范围损失)
    """
    def masked_loss(pred: torch.Tensor, target: torch.Tensor, padding_mask: torch.BoolTensor, criterion: Callable[..., torch.Tensor]):
        "计算掩码区域的损失，仅对有效帧求平均"
        # 计算逐元素损失 [batch_size, seq_len, ...]
        elementwise_loss = criterion(pred, target, reduction="none")

        # 重塑损失张量以便统一 variance 和 piano_roll 掩码应用逻辑
        # [batch_size, seq_len, feature_dim]，其中 feature_dim 可能为 1 或 128
        loss_reshaped = elementwise_loss.view(*elementwise_loss.shape[:2], -1)

        # 扩展掩码以匹配损失张量的形状
        expanded_mask = padding_mask.unsqueeze(2).expand_as(loss_reshaped)

        # 将填充区域的损失置零
        masked_loss = loss_reshaped.masked_fill(expanded_mask, 0)

        # 计算损失平均值
        return masked_loss.sum(dim=[1, 2]) / (~expanded_mask).sum(dim=[1, 2])

    # 计算各分量损失
    piano_roll_loss = masked_loss(piano_roll_pred, piano_roll_target.to(dtype=piano_roll_pred.dtype), padding_mask, F.binary_cross_entropy_with_logits)
    note_counts_loss = masked_loss(note_counts_pred, note_counts_target, padding_mask, F.mse_loss)
    pitch_mean_loss = masked_loss(pitch_mean_pred, pitch_mean_target, padding_mask, F.mse_loss)
    pitch_range_loss = masked_loss(pitch_range_pred, pitch_range_target, padding_mask, F.mse_loss)

    # 返回各分量损失
    return piano_roll_loss, note_counts_loss, pitch_mean_loss, pitch_range_loss


def train(
    model: MidiNet,
    dataloader: DataLoader,
    optimizer: optim.AdamW,
    scaler: GradScaler,
    completed_iters: int,
    writer: SummaryWriter,
    accumulation_steps: int = 1,
    device: torch.device = torch.device("cpu")
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    训练 MidiNet 模型的函数，支持梯度累积和混合精度训练

    Args:
        model: 待训练的 MidiNet 模型实例
        dataloader: 训练数据加载器
        optimizer: 模型优化器
        scaler: 混合精度梯度缩放器
        completed_iters: 已经完成多少次迭代，记录损失用
        writer: 记录训练损失用的
        accumulation_steps: 梯度累积步数
        device: 训练设备（默认使用 CPU）

    Returns:
        最后一步的钢琴卷帘、音符数量、平均音高、音高范围和目标

    Examples:
        >>> losses = train(model, loader, opt)
        >>> plt.plot(losses)  # 绘制损失曲线
    """
    # 设置模型为训练模式
    model.train()

    # 计算前向传播多少次
    num_steps = len(dataloader) // accumulation_steps * accumulation_steps

    # 创建进度条，显示训练进度
    progress_bar = tqdm(total=num_steps, desc="Train")

    # 当前累积步骤的总损失
    acc_piano_roll_loss = acc_note_counts_loss = acc_pitch_means_loss = acc_pitch_ranges_loss = 0

    # 提前清空梯度
    optimizer.zero_grad()

    # 遍历整个训练集
    for step, batch in zip(range(num_steps), dataloader):
        # 数据移至目标设备
        piano_roll, note_counts, pitch_means, pitch_ranges, padding_mask = [item.to(device=device) for item in batch]

        # 自动混合精度环境
        with autocast(device.type, dtype=torch.float16):
            piano_roll_pred, note_counts_pred, pitch_means_pred, pitch_ranges_pred, _ = model(piano_roll[:, :-1], note_counts, pitch_means, pitch_ranges, padding_mask[:, :-1])  # 模型前向传播（使用教师强制）
            all_loss = midinet_loss(piano_roll_pred, note_counts_pred, pitch_means_pred, pitch_ranges_pred, piano_roll[:, 1:], note_counts, pitch_means, pitch_ranges, padding_mask[:, 1:])  # 计算损失
            piano_roll_loss, note_counts_loss, pitch_means_loss, pitch_ranges_loss = (loss.mean() for loss in all_loss)  # 计算整个批次的损失
            value = piano_roll_loss + note_counts_loss + pitch_means_loss + pitch_ranges_loss

        # 梯度缩放与反向传播
        scaler.scale(value).backward()

        # 更新累积损失
        acc_piano_roll_loss += piano_roll_loss.item() / accumulation_steps
        acc_note_counts_loss += note_counts_loss.item() / accumulation_steps
        acc_pitch_means_loss += pitch_means_loss.item() / accumulation_steps
        acc_pitch_ranges_loss += pitch_ranges_loss.item() / accumulation_steps

        # 达到累积步数时更新参数
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # 更新模型参数
            scaler.update()  # 调整缩放因子
            optimizer.zero_grad()  # 清空梯度
            global_step = completed_iters + ((step + 1) // accumulation_steps) - 1  # 计算全局步数

            # 记录损失
            for loss_name, loss_value in [
                ("Piano Roll", acc_piano_roll_loss),
                ("Note Count", acc_note_counts_loss),
                ("Pitch Mean", acc_pitch_means_loss),
                ("Pitch Range", acc_pitch_ranges_loss),
            ]:
                writer.add_scalars(f"Loss/{loss_name}", {"Train": loss_value}, global_step)

            # 重置累积损失
            acc_piano_roll_loss = acc_note_counts_loss = acc_pitch_means_loss = acc_pitch_ranges_loss = 0

        # 更新进度条
        progress_bar.update()

        # 最后一步时，记录训练集上预测和目标值
        if step + 1 == num_steps:
            results = [
                x[0].detach().cpu().numpy()
                for x in [F.sigmoid(piano_roll_pred), piano_roll[:, 1:], note_counts_pred, note_counts, pitch_means_pred, pitch_means, pitch_ranges_pred, pitch_ranges]
            ]
            return results[::2], results[1::2]


@torch.inference_mode
def validate(
    model: MidiNet,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu")
) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], list[tuple[float, float, float, float]]]:
    """
    在验证集上评估 MidiNet 模型的性能，计算模型在验证集上的各项损失值
    使用推理模式禁用梯度计算，节省内存并加速验证过程
    支持自动混合精度计算，在保持精度的同时提升计算效率

    Args:
        model: 要验证的 MidiNet 模型实例
        dataloader: 验证集的数据加载器，提供批次数据
        device: 计算设备，用于指定模型和数据所在的硬件设备

    Returns:
        tuple[tuple[预测, 目标], list[tuple[钢琴卷帘损失, 音符数量损失, 音高均值损失, 音高范围损失]]]

    Examples:
        >>> validation_results = validate(model, val_loader, 1.0, torch.device("cuda"))
        >>> piano_roll_loss_avg = sum(result[1][0] for result in validation_results) / len(validation_results)
    """
    # 设置模型为评估模式
    model.eval()

    # 初始化损失列表和预测-目标
    loss_results = []
    logged_pred = logged_target = None

    # 遍历验证集所有批次数据，显示进度条
    for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc="Validate")):
        # 数据移至目标设备
        piano_roll, note_counts, pitch_means, pitch_ranges, padding_mask = [item.to(device=device) for item in batch]

        # 自动混合精度环境
        with autocast(device.type, dtype=torch.float16):
            # 模型前向传播（不使用教师强制）
            piano_roll_pred, note_counts_pred, pitch_means_pred, pitch_ranges_pred, _ = model(piano_roll[:, :-1], padding_mask=padding_mask[:, :-1])

            # 计算损失
            all_loss = midinet_loss(piano_roll_pred, note_counts_pred, pitch_means_pred, pitch_ranges_pred, piano_roll[:, 1:], note_counts, pitch_means, pitch_ranges, padding_mask[:, 1:])

        # 记录当前批次的损失信息
        for batch_idx in range(piano_roll.size(0)):
            loss_results.append(tuple(loss[batch_idx].item() for loss in all_loss))

        # 如果没有记录任何预测，并且随机数大于 0.5 或者已经是最后一个批次，则记录目标和损失
        if logged_pred is None and (random.random() > 0.5 or batch_idx == len(dataloader) - 1):
            results = [
                x[0].cpu().numpy()
                for x in [F.sigmoid(piano_roll_pred), piano_roll[:, 1:], note_counts_pred, note_counts, pitch_means_pred, pitch_means, pitch_ranges_pred, pitch_ranges]
            ]
            logged_pred, logged_target = results[::2], results[1::2]

    return (logged_pred, logged_target), loss_results


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="训练 TkTTS 模型")
    parser.add_argument("num_epochs", type=int, help="训练的总轮数")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="加载和保存检查点的路径")
    parser.add_argument("-t", "--train-dataset", type=pathlib.Path, required=True, help="训练集文件路径")
    parser.add_argument("-v", "--val-dataset", type=pathlib.Path, required=True, help="验证集文件路径")
    parser.add_argument("-tt", "--train-max-batch-tokens", default=2048, type=int, help="训练时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-tv", "--val-max-batch-tokens", default=4096, type=int, help="验证时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-lr", "--learning-rate", default=DEFAULT_LEARNING_RATE, type=float, help="学习率，默认为 %(default)s")
    parser.add_argument("-wd", "--weight-decay", default=DEFAULT_WEIGHT_DECAY, type=float, help="权重衰减系数，默认为 %(default)s")
    parser.add_argument("-de", "--encoder-dropout", default=DEFAULT_ENCODER_DROPOUT, type=float, help="编码器 Dropout 概率，默认为 %(default)s")
    parser.add_argument("-dd", "--decoder-dropout", default=DEFAULT_DECODER_DROPOUT, type=float, help="解码器 Dropout 概率，默认为 %(default)s")
    parser.add_argument("-dv", "--variance-predictor-dropout", default=DEFAULT_VARIANCE_PREDICTOR_DROPOUT, type=float, help="变异性预测器 Dropout 概率，默认为 %(default)s")
    parser.add_argument("-dp", "--pitch-dropout", default=DEFAULT_PITCH_DROPOUT, type=float, help="音高特征编码器 Dropout 概率，默认为 %(default)s")
    parser.add_argument("-as", "--accumulation-steps", default=DEFAULT_ACCUMULATION_STEPS, type=int, help="梯度累积步数，默认为 %(default)s")
    parser.add_argument("-pr", "--piano-roll-length", default=DEFAULT_PIANO_ROLL_LENGTH, type=int, help="记录预测-目标钢琴卷帘时，最大允许的长度，超过该长度的钢琴卷帘将会被截取，默认为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 设置当前进程的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取检查点
    print("读取检查点 ...")
    model_state, model_config, ckpt_info, optimizer_state = load_checkpoint_train(args.ckpt_path)

    # 创建模型并加载状态
    model = MidiNet(model_config, args.pitch_dropout, args.encoder_dropout, args.decoder_dropout, args.variance_predictor_dropout)
    model.load_state_dict(model_state)

    # 转移模型到设备
    model = model.to(device)

    # 创建优化器并加载状态
    optimizer = optim.AdamW(model.parameters())
    optimizer.load_state_dict(optimizer_state)

    # 设置学习率、权重衰减系数
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
        group["weight_decay"] = args.weight_decay

    # 创建混合精度梯度缩放器
    scaler = GradScaler(device)

    # 加载训练数据集
    print("加载训练集 ...")
    train_dataset = MidiDataset(args.train_dataset)
    train_sampler = MidiDatasetSampler(train_dataset, args.train_max_batch_tokens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=sequence_collate_fn, num_workers=0)

    # 加载验证数据集
    print("加载验证集 ...")
    val_dataset = MidiDataset(args.val_dataset)
    val_sampler = MidiDatasetSampler(val_dataset, args.val_max_batch_tokens)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=sequence_collate_fn, num_workers=0)

    # 创建一个 SummaryWriter 实例，用于记录训练过程中的指标和可视化数据
    writer = SummaryWriter(args.ckpt_path / f"logdir/default")

    # 开始训练
    for epoch in range(args.num_epochs):
        # 计算累积 Epoch 数
        current_epoch = ckpt_info["completed_epochs"] + epoch

        # 训练一轮模型
        train_sampler.set_epoch(current_epoch)
        train_pred, train_target = train(model, train_loader, optimizer, scaler, len(train_loader) // args.accumulation_steps * current_epoch, writer, args.accumulation_steps, device=device)

        # 验证模型效果
        val_sampler.set_epoch(current_epoch)
        (val_pred, val_target), val_loss = validate(model, val_loader, device)

        # 选择训练集最后一步预测，和验证集随机选择结果，绘制预测钢琴卷帘图和其目标钢琴卷帘图
        for title, pred, target in [
            ("Train Sample", train_pred, train_target),
            ("Validate Sample", val_pred, val_target),
        ]:
            # 创建图像和坐标轴
            figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

            # 绘制对比图
            for ax, figure_title, piano_roll, _, pitch_mean, pitch_range in [
                (ax1, "Predicted Piano Roll", *pred),
                (ax2, "True Piano Roll", *target)
            ]:
                # 设置标题
                ax.set_title(figure_title)

                # 随机选取长度截断钢琴卷帘
                if len(piano_roll) > args.piano_roll_length:
                    start_pos = random.randint(0, len(piano_roll) - args.piano_roll_length)
                    piano_roll, pitch_mean, pitch_range = [x[start_pos:start_pos + args.piano_roll_length] for x in [piano_roll, pitch_mean, pitch_range]]

                # 绘制钢琴卷帘
                plot_piano_roll(piano_roll, pitch_mean, pitch_range, ax)

            # 添加图像到 writer
            writer.add_figure(f"Epoch {current_epoch + 1}/{title}", figure)

        # 绘制验证损失分布直方图，记录验证损失
        for loss_idx, loss_name in enumerate(["Piano Roll", "Note Count", "Pitch Mean", "Pitch Range"]):
            loss_values = [all_loss[loss_idx] for all_loss in val_loss]
            writer.add_histogram(f"Epoch {current_epoch + 1}/Validate/{loss_name} Loss Distribution", np.array(loss_values))
            writer.add_scalars(f"Loss/{loss_name}", {"Valid": np.array(loss_values).mean()}, len(train_loader) // args.accumulation_steps * (current_epoch + 1))

    # 关闭 SummaryWriter 实例，确保所有记录的数据被写入磁盘并释放资源
    writer.close()

    # 保存当前模型的检查点
    ckpt_info["completed_epochs"] += args.num_epochs
    save_checkpoint(
        args.ckpt_path,
        model.cpu().state_dict(),
        optimizer.state_dict(),
        ckpt_info
    )
    print(f"训练完成，模型已保存到 {args.ckpt_path}，训练过程记录保存到 {args.ckpt_path / 'logdir'}，你可以通过 `tensorboard --logdir {args.ckpt_path / 'logdir'}` 查看。")


if __name__ == "__main__":
    main(parse_args())
