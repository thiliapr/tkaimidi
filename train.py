# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import random
import pathlib
import argparse
from typing import Optional
from collections.abc import Iterator
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from utils.checkpoint import load_checkpoint_train, save_checkpoint
from utils.constants import DEFAULT_ACCUMULATION_STEPS, DEFAULT_PROBABILITY_MAPS_LENGTH, DEFAULT_TRAIN_MAX_BATCH_TOKENS, DEFAULT_VAL_MAX_BATCH_TOKENS, DEFAULT_DROPOUT, PITCH_RANGE
from utils.model import GPT

# 解除线程数量限制
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())


class MidiDataset(Dataset):
    """
    MIDI 数据集加载器

    Args:
        dataset_file: 快速训练数据集文件
        pitch_perturb_prob: 随机替换音高的最大概率因子

    Yields:
        输入和标签序列对
    """

    def __init__(self, dataset_file: os.PathLike, pitch_perturb_prob: float):
        # 获取所有旋律
        self.data_samples = np.load(dataset_file)
        self.pitch_perturb_prob = pitch_perturb_prob

    def get_lengths(self) -> list[int]:
        return self.data_samples["length"].tolist()

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        sequence = self.data_samples[f"{index}"]
        input_sequence = sequence[:-1].copy()  # 输入序列: 从开始到倒数第二个音符
        target_sequence = sequence[1:].copy()  # 目标序列: 从第二个音符到最后一个音符

        # 动态调整音高扰动概率
        current_perturb_prob = random.random() * self.pitch_perturb_prob

        # 模拟瞎几把乱拽音符，但保持旋律整体结构
        for idx in range(len(input_sequence)):
            if random.random() < current_perturb_prob:
                input_pitch = input_sequence[idx]
                target_pitch = target_sequence[idx]

                # 计算可移动的音高范围，确保被替换的音高和下一个音高都在合法范围内
                lower_bound = -min(input_pitch, PITCH_RANGE * 2 - target_pitch)
                upper_bound = min(PITCH_RANGE * 2 - input_pitch, target_pitch)

                # 随机生成音高扰动值
                pitch_perturbation = random.randint(lower_bound, upper_bound)

                # 在当前位置拽动音符，改变音高
                input_sequence[idx] += pitch_perturbation

                # 调整目标序列，告诉模型虽然这里被拽乱了，但下一个音符应该回到正轨
                target_sequence[idx] -= pitch_perturbation

                # 调整下一个输入，保持后续旋律的连贯过渡
                if idx + 1 < len(input_sequence):
                    input_sequence[idx + 1] -= pitch_perturbation

        return input_sequence, target_sequence

    def __len__(self) -> int:
        # 这个文件储存了每个序列的长度，所以用总数减去 1 就是样本数
        return len(self.data_samples) - 1


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
        length = dataset.get_lengths()
        self.index_and_lengths = [
            (idx, length[idx])
            for idx in range(len(dataset))
        ]

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


def sequence_collate_fn(batch: list[tuple[list[int], list[int]]]) -> tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
    """
    处理变长序列数据的批次整理函数
    将输入的多个变长序列样本整理为批次张量，并生成相应的填充掩码和序列长度信息

    工作流程：
    1. 解压批次数据并将每个特征转换为 PyTorch 张量
    2. 创建填充掩码
    3. 计算序列的实际长度
    4. 对所有序列进行填充对齐处理
    5. 返回整理后的批次数据

    Args:
        batch: 包含多个样本的列表

    Returns:
        包含整理后批次数据的元组，包括填充后的旋律序列合填充掩码

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> dataset = MidiDataset("dataset.npz", pitch_perturb_prob=0.1)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=sequence_collate_fn)
        >>> for inputs, labels, padding_mask in dataloader:
        >>>     logits = model(inputs, padding_mask=padding_mask[:, :-1])
        >>>     loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view(labels.shape).masked_fill(padding_mask[:, 1:], 0).sum() / (~padding_mask[:, 1:]).sum()
    """
    # 解压批次数据并转换为张量
    inputs, labels = [[torch.tensor(sequence, dtype=torch.long) for sequence in todoname2] for todoname2 in zip(*batch)]

    # 创建填充掩码用于标识有效数据位置
    max_length = max(len(seq) for seq in inputs) + 1
    sequence_lengths = torch.tensor([len(seq) + 1 for seq in inputs])
    padding_mask = torch.arange(max_length).unsqueeze(0) >= sequence_lengths.unsqueeze(1)

    # 对变长序列进行填充对齐，使批次内所有样本长度一致
    inputs, labels = [torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True) for sequences in [inputs, labels]]

    # 返回批次数据
    return inputs, labels, padding_mask


@torch.inference_mode()
def validate(
    model: GPT,
    dataloader: DataLoader,
    device: torch.device = "cpu"
) -> tuple[tuple[torch.Tensor, torch.Tensor], list[float]]:
    """
    在验证集上评估 MidiNet 模型的性能，计算模型在验证集上的损失值
    使用推理模式禁用梯度计算，节省内存并加速验证过程
    支持自动混合精度计算，在保持精度的同时提升计算效率

    Args:
        model: 要验证的 MidiNet 模型实例
        dataloader: 验证集的数据加载器，提供批次数据
        device: 计算设备，用于指定模型和数据所在的硬件设备

    Returns:
        tuple[tuple[预测, 目标], list[每个样本的损失]]

    Examples:
        >>> validation_results = validate(model, val_loader, "cuda")
        >>> loss_avg = sum(result[1] for result in validation_results) / len(validation_results)
    """
    # 设置模型为评估模式
    model.eval()

    # 初始化损失列表和预测-目标
    loss_results = []
    logged_pred = logged_target = None

    # 遍历验证集所有批次数据，显示进度条
    for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc="Validate")):
        # 数据移至目标设备
        inputs, labels, padding_mask = [item.to(device=device) for item in batch]

        # 自动混合精度环境
        with autocast(device, dtype=torch.float16):
            # 模型前向传播（不使用教师强制）
            logits, _ = model(inputs, padding_mask=padding_mask[:, :-1])

            # 计算损失
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").reshape(labels.shape).masked_fill(padding_mask[:, 1:], 0).sum(dim=1) / (~padding_mask[:, 1:]).sum(dim=1)

        # 记录当前批次的损失信息
        loss_results.extend(loss.tolist())

        # 如果没有记录任何预测，并且抽选到该批次或者已经是最后一个批次，则记录该批次的预测-目标
        if logged_pred is None and (random.randint(0, len(dataloader) - 1) == 0 or batch_idx == len(dataloader) - 1):
            logged_pred, logged_target = logits[0], labels[0]

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
    parser.add_argument("num_val_cycles", type=int, help="训练过程中进行多少次验证")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="加载和保存检查点的路径")
    parser.add_argument("-t", "--train-dataset", type=pathlib.Path, required=True, help="训练集文件路径")
    parser.add_argument("-v", "--val-dataset", type=pathlib.Path, required=True, help="验证集文件路径")
    parser.add_argument("-tt", "--train-max-batch-tokens", default=DEFAULT_TRAIN_MAX_BATCH_TOKENS, type=int, help="训练时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-tv", "--val-max-batch-tokens", default=DEFAULT_VAL_MAX_BATCH_TOKENS, type=int, help="验证时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-pp", "--pitch-perturb-prob", default=0.2, type=float, help="输入序列中每个音符被随机替换音高的最大概率因子，默认为 %(default)s")
    parser.add_argument("-vp", "--val-per-steps", default=1024, type=int, help="每训练多少步进行一次验证，默认为 %(default)s")
    parser.add_argument("-dr", "--dropout", default=DEFAULT_DROPOUT, type=float, help="Dropout 概率，默认为 %(default)s")
    parser.add_argument("-as", "--accumulation-steps", default=DEFAULT_ACCUMULATION_STEPS, type=int, help="梯度累积步数，默认为 %(default)s")
    parser.add_argument("-pm", "--probability-maps-length", default=DEFAULT_PROBABILITY_MAPS_LENGTH, type=int, help="记录预测-目标对比时，最大允许的长度，超过该长度将会被截取，默认为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 设置当前进程的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取检查点
    print("读取检查点 ...")
    model_state, model_config, ckpt_info, optimizer_state, scaler_state = load_checkpoint_train(args.ckpt_path)

    # 创建模型并加载状态
    model = GPT(model_config, args.dropout)
    model.load_state_dict(model_state)

    # 转移模型到设备
    model = model.to(device)

    # 创建优化器并加载状态
    optimizer = optim.AdamW(model.parameters())
    optimizer.load_state_dict(optimizer_state)

    # 创建混合精度梯度缩放器并加载状态
    scaler = GradScaler(device)
    scaler.load_state_dict(scaler_state)

    # 加载训练数据集
    print("加载训练集 ...")
    train_dataset = MidiDataset(args.train_dataset, args.pitch_perturb_prob)
    train_sampler = MidiDatasetSampler(train_dataset, args.train_max_batch_tokens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=sequence_collate_fn, num_workers=0)

    # 加载验证数据集
    print("加载验证集 ...")
    val_dataset = MidiDataset(args.val_dataset, 0)
    val_sampler = MidiDatasetSampler(val_dataset, args.val_max_batch_tokens)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=sequence_collate_fn, num_workers=0)

    # 创建一个 SummaryWriter 实例，用于记录训练过程中的指标和可视化数据
    writer = SummaryWriter(args.ckpt_path / f"logdir")

    # 开始训练
    optimizer.zero_grad()  # 提前清零梯度
    model.train()  # 设置模型为训练模式
    progress_bar = tqdm(total=args.val_per_steps * args.num_val_cycles * args.accumulation_steps, desc="Train")
    current_steps = ckpt_info["completed_steps"]
    acc_loss = 0
    torch.autograd.set_detect_anomaly(True) 
    while True:
        if current_steps - ckpt_info["completed_steps"] >= args.val_per_steps * args.num_val_cycles * args.accumulation_steps:
            break

        # 为当前 epoch 设置采样器
        train_sampler.set_epoch(current_steps)
        val_sampler.set_epoch(current_steps)

        for batch in train_loader:
            if current_steps - ckpt_info["completed_steps"] >= args.val_per_steps * args.num_val_cycles * args.accumulation_steps:
                break

            # 将数据移至目标设备
            inputs, labels, padding_mask = [item.to(device=device) for item in batch]

            # 自动混合精度环境
            with autocast(device, dtype=torch.float16):
                # 模型前向传播，使用教师强制
                logits, _ = model(inputs, padding_mask=padding_mask[:, :-1])

                # 计算损失
                loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").reshape(labels.shape).masked_fill(padding_mask[:, 1:], 0).sum() / (~padding_mask[:, 1:]).sum()

                # 梯度累积
                loss = loss / args.accumulation_steps
                acc_loss += loss.item()

            # 反向传播
            scaler.scale(loss).backward()

            # 进度条更新
            progress_bar.update()

            # 每隔一定步数更新一次参数
            current_steps += 1
            if current_steps % args.accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # 记录训练指标
                writer.add_scalar("Loss/Train", acc_loss, current_steps // args.accumulation_steps - 1)
                writer.add_scalar("GradNorm/Train", grad_norm.item(), current_steps // args.accumulation_steps - 1)
                acc_loss = 0

                # 记录每层的缩放因子
                for layer_idx, layer in enumerate(model.layers):
                    writer.add_scalar(f"Scale/Layer #{layer_idx} Feedforward", layer.feedforward_scale.item(), current_steps // args.accumulation_steps)
                    writer.add_scalar(f"Scale/Layer #{layer_idx} Attention", layer.attention_scale.item(), current_steps // args.accumulation_steps)

            # 每隔一定步数进行验证
            if current_steps % (args.accumulation_steps * args.val_per_steps) == 0:
                (val_pred, val_target), val_losses = validate(model, val_loader, device)
                model.train()  # 切换回训练模式
                optimizer.zero_grad()  # 清零梯度

                # 记录验证损失
                val_loss_avg = sum(val_losses) / len(val_losses)
                writer.add_scalar("Loss/Validate", val_loss_avg, current_steps // args.accumulation_steps)

                # 记录预测-目标对比图
                for pred, target, result_type in [(logits[0], labels[0], "Train"), (val_pred, val_target, "Validate")]:
                    # 处理长度过长的情况
                    pred = F.softmax(pred.detach().cpu()[:args.probability_maps_length], dim=-1)
                    target = F.one_hot(target.detach().cpu()[:args.probability_maps_length], num_classes=pred.size(-1))

                    # 生成对比图
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                    ax1.set_title("Predicted Probability Maps")
                    ax2.set_title("Difference (Target - Predicted)")
                    for image, ax in [(pred, ax1), ((target - pred).abs(), ax2)]:
                        ax.set_xlabel("Time Step")
                        ax.set_ylabel("Relative Pitch")
                        plt.colorbar(ax.imshow(image.T, aspect="auto", origin="lower", cmap="viridis", extent=[0, pred.size(0), -0.5 - pred.size(-1) // 2, pred.size(-1) // 2 + 0.5]), ax=ax)
                    writer.add_figure(f"Prediction vs Target/{result_type} Iteration #{current_steps // args.accumulation_steps}", fig, current_steps // args.accumulation_steps)
                    plt.close(fig)

    # 关闭进度条和 SummaryWriter
    progress_bar.close()
    writer.close()

    # 保存当前模型的检查点
    ckpt_info["completed_steps"] = current_steps
    save_checkpoint(
        args.ckpt_path,
        model.cpu().state_dict(),
        optimizer.state_dict(),
        scaler.state_dict(),
        ckpt_info
    )
    print(f"训练完成，模型已保存到 {args.ckpt_path}，训练过程记录保存到 {args.ckpt_path / 'logdir'}，你可以通过 `tensorboard --logdir {args.ckpt_path / 'logdir'}` 查看。")


if __name__ == "__main__":
    main(parse_args())
