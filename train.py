# MIDI 音乐生成模型训练模块
"""
MIDI 音乐生成模型训练模块

本模块提供完整的MIDI音乐生成模型训练流程，包含数据集处理、模型训练、验证评估及可视化功能

特性:
- 高效内存管理: 支持TB级MIDI数据处理
- 容错机制: 自动回退异常训练状态
- 可重复性: 通过生成器状态保存实现实验复现
- Flooding Loss: 训练损失低于某值时执行梯度上升，有利于提高泛化性能。论文见: https://arxiv.org/abs/2002.08709

使用示例:
>>> from train import main
>>> main()  # 启动完整训练流程

配置参数说明见main()函数中的config字典
"""

# Copyright (C)  thiliapr 2024-2025
# License: AGPLv3-or-later

import sys
import copy
import math
import random
import shutil
import pathlib
import tempfile
import mido
import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from functools import partial

import warnings
import os
from multiprocessing import cpu_count

# 忽略警告
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*deprecated.*")
# 解除线程数量限制
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count())
torch.set_num_threads(cpu_count())

# 在非Jupyter环境下导入模型和工具库
if "get_ipython" not in globals():
    from model import MidiNet, TIME_PRECISION, MAX_NOTE, VOCAB_SIZE, load_checkpoint, save_checkpoint
    from utils import midi_to_notes, normalize_times, notes_to_note_intervals, empty_cache


class MidiDataset(Dataset):
    """
    MIDI 数据集类，用于加载和处理MIDI文件生成训练序列

    特性:
    - 自动处理不同长度的MIDI文件
    - 支持音符平移和时间缩放
    - 自动填充短序列，保证数据可用性

    Args:
        path: MIDI文件存储路径（支持嵌套目录结构）
        seq_size: 输入序列长度 + 1
        interval_repeat: 音符过少重复时，两段音乐的时间间隔
        show_progress: 显示进度条
    """

    def __init__(self, path: pathlib.Path, seq_size: int, interval_repeat: int = 8, show_progress: bool = True):
        self.data: list = []  # 存储每个 MIDI 文件的数据
        self.seq_size = seq_size  # 基础序列长度 (用于模型输入的上下文窗口)
        self.length = 0  # 经过数据增强后的总样本数

        # 遍历目录获取MIDI文件 (按文件名排序保证可重复性)
        # midi_files = sorted(list(path.glob("**/*.mid")), key=lambda x: x.name)
        midi_files = list(filter(lambda x: x.name.startswith("Touhou"), path.glob("**/*.mid")))

        if show_progress:
            progress_bar = tqdm.tqdm(desc="Load Dataset", total=len(midi_files))

        for filepath in midi_files:
            # 解析 MIDI 文件，提取音符信息
            parsed_data = [(note, start_at) for _, note, start_at, _ in midi_to_notes(mido.MidiFile(filepath, clip=True))]

            # 使用两种时间归一化模式 (严格/宽松) 增加数据多样性
            for param_strict in [False, True]:
                # 规范化时间
                data = normalize_times(parsed_data, TIME_PRECISION, strict=param_strict)

                # 转换格式
                data = notes_to_note_intervals(data, -1)

                # 通过循环填充达到最小长度要求
                if len(data) < seq_size:
                    data += [-1] * interval_repeat
                    data *= seq_size // len(data) + 1  # 重复数据，确保序列长度足够
                    data = data[:-interval_repeat]

                # 计算每个子序列的数据增强潜力
                offsets: list[int] = []
                for i in range(len(data) - seq_size):
                    # 提取当前窗口的序列数据
                    seq = data[i:i + seq_size]
                    if seq[0] == -1:
                        continue  # 跳过以间隔为开头的序列

                    # 提取序列的音高
                    pitches = [pitch for pitch in seq if pitch != -1]

                    # 平移至最小音符为0 (保留相对音高关系)
                    min_pitch = min(pitches)
                    pitches = [pitch - min_pitch for pitch in pitches]

                    # 通过八度移调保持在 [0, MAX_NOTE] 范围内
                    pitches = [
                        (pitch - math.ceil((pitch + 1 - MAX_NOTE) / 12) * 12) if (pitch > MAX_NOTE) else pitch
                        for pitch in pitches
                    ]

                    # 计算可平移的音符偏移量 (数据增强潜力)
                    note_offsets = MAX_NOTE - max(pitches) + 1  # 剩余可上移空间 (包括原来的空间, 所以加1)

                    offsets.append(note_offsets)
                    self.length += note_offsets  # 累加增强后的样本数

                # 存储处理后的数据, 每个文件保存两种时间归一化版本
                self.data.append((offsets, data))

            if show_progress:
                progress_bar.update()

        if show_progress:
            progress_bar.close()

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        """
        索引访问实现逻辑:
        1. 遍历所有文件数据
        2. 在单个文件内遍历所有序列
        3. 通过偏移量计算定位具体增强参数
        4. 应用数据增强生成最终序列
        """
        for offsets, filedata in self.data:
            found = False
            # 通过递减法找到目标所在的序列
            seq_offsets_index = 0
            for seq_index in range(len(filedata) - self.seq_size):
                if filedata[seq_index] == -1:
                    continue  # 跳过以间隔为开头的序列

                if index >= offsets[seq_offsets_index]:
                    index -= offsets[seq_offsets_index]  # 不在当前序列范围，调整剩余索引
                else:
                    note_offset = index
                    found = True
                    break

                seq_offsets_index += 1
            if found:
                # 提取原始序列数据
                seq = filedata[seq_index:seq_index + self.seq_size]
                break
        if not found:
            raise Exception(f"404 Not Found: {index}")

        # 音符下移至最低
        min_note = min(note for note in seq if note != -1)
        seq = [note if (note == -1) else (note - min_note) for note in seq]

        # 音高越界处理 (确保在范围内)
        seq = [note - math.ceil((note + 1 - MAX_NOTE) / 12) * 12 if note > MAX_NOTE else note for note in seq]

        # 应用音符偏移 (数据增强)
        seq = [note if (note == -1) else (note + note_offset) for note in seq]

        # 将 -1 替换为模型输入
        seq = [MAX_NOTE + 1 if interval == -1 else interval for interval in seq]

        # 返回输入和目标序列
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)


class SplitDataset(Dataset):
    """
    数据集包装器，允许将给定数据集拆分为从指定索引开始的子集

    Args:
        dataset: 原始数据集
        data_start: 拆分的起始索引
        length: 拆分数据集的长度
    """

    def __init__(self, dataset: Dataset, data_start: int, length: int):
        self.data_start = data_start  # 拆分的起始索引
        self.length = length  # 拆分数据集的长度
        self.dataset = dataset  # 原始数据集

    def __len__(self):
        return self.length

    def __getitem__(self, i: int):
        actual_index = self.data_start + i  # 计算在原始数据集中的实际索引
        if actual_index >= len(self.dataset):  # 如果索引超过原始数据集的长度，则进行循环处理
            actual_index -= len(self.dataset)
        return self.dataset[actual_index]


def split_dataset(dataset: Dataset, train_length: float, train_start: int):
    """
    将数据集拆分为训练集和验证集。

    Args:
        dataset: 原始数据集
        train_length: 用于训练的数据集比例（介于 0 和 1 之间）
        train_start: 分割数据集时训练集的开始索引

    Returns:
        包含训练集和验证集的元组。
    """
    train_length = int(train_length * len(dataset))  # 计算训练样本的数量
    val_length = len(dataset) - train_length  # 计算验证样本的数量
    val_start = train_start + train_length  # 计算验证数据集的起始索引
    if val_start > len(dataset):  # 如果验证起始索引超过原始数据集的长度，则进行循环处理
        val_start -= len(dataset)
    return SplitDataset(dataset, train_start, train_length), SplitDataset(dataset, val_start, val_length)  # 创建并返回训练集和验证集


def train(
    model: MidiNet,
    dataset: MidiDataset,
    optimizer: optim.Adam,
    train_batch_size: int,
    val_batch_size: int,
    train_length: float = 0.8,
    val_steps: int = 256,
    val_per_step: int = 4096,
    steps_to_val: int = 256,
    flooding_level: float = 0,
    train_start: int = 0,
    generator_state: torch.tensor = ...,
    last_batch: int = 0,
    device: torch.device = torch.device("cpu")
) -> tuple[list[float], list[float], list[float], list[float], int, torch.tensor]:
    """
    训练模型并记录训练和验证的损失。

    Args:
        model: 要训练的模型
        dataset: 用于训练的 MIDI 数据集
        optimizer: 优化器，用于更新模型参数
        train_batch_size: 每个训练批次的样本数
        val_batch_size: 每个验证批次的样本数
        train_length: 训练集占数据集的比例
        val_steps: 进行多少次验证，决定了训练过程中训练的步数 (train_steps = val_steps * val_per_step)
        val_per_step: 每多少个训练步骤后进行一次验证，决定了训练过程中验证的频率
        steps_to_val: 每次验证时验证多少个批次，决定了验证时使用的数据量
        flooding_level: 允许训练最低的损失值。训练损失低于该值则执行梯度上升，否则执行梯度下降
        train_start: 拆分数据集时训练集的开始索引
        generator_state: 训练数据的索引随机采样器的生成器的状态
        last_batch: 上次训练时的未训练完成的epoch训练了多少个batch
        device: 用于训练的设备

    Returns:
        训练、验证损失和准确率的历史记录和生成器状态和最后训练的批次
    """
    empty_cache()  # 清理缓存以释放内存

    # 根据 train_length 切分训练集和验证集
    train_dataset, val_dataset = split_dataset(dataset, train_length, train_start)
    steps_to_val = min(steps_to_val, len(val_dataset) // val_batch_size)  # 确保验证步骤不超过验证集大小

    # 获取训练数据加载器
    train_generator = torch.Generator()
    if generator_state != ...:
        train_generator.set_state(generator_state)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, generator=train_generator)

    if not torch.cuda.is_available():
        print(f"train(): 使用 {cpu_count()} 个 CPU 核心进行训练。")

    # 将模型移动到设备
    model = model.to(device)

    # 检查是否使用多GPU
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # 使用 DataParallel 进行多GPU训练

    # 初始化记录
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    # 初始化最后正常参数
    last_normal_state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

    # 创建进度条，显示训练进度
    progress_bar = tqdm.tqdm(desc="Training", total=val_steps * (val_per_step + steps_to_val))

    step = 0
    model.train()  # 确保模型在训练模式下
    train_begin = True  # 刚开始训练
    epoch_train_loss, epoch_train_acc = [], []  # 一次验证中训练的损失、准确率，用于计算训练损失、准确率标准差
    while step < val_steps * val_per_step:  # 训练直到达到验证步骤上限
        generator_state = train_generator.get_state()  # 获取当前生成器状态
        loader_iter = iter(train_loader)  # 获取该 epoch 的 DataloderIter 对象

        # 将 iter 调整到上次训练结束时的状态
        last_batch %= len(train_loader)
        if train_begin:
            for _ in range(last_batch):
                next(loader_iter._sampler_iter)
            train_begin = False

        for inputs, labels in loader_iter:
            step += 1  # 增加步骤计数
            last_batch += 1  # 增加批次计数

            # 训练阶段
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            optimizer.zero_grad()  # 清空优化器中的梯度
            outputs = model(inputs).view(-1, VOCAB_SIZE)  # 前向传播
            loss = F.cross_entropy(outputs, labels)  # 计算交叉熵损失
            flood = (loss - flooding_level).abs() + flooding_level  # 梯度下降或上升

            # 检查损失是否为 NaN
            if torch.isnan(loss):
                unsaved_steps = step % val_per_step
                progress_bar.update(1 - unsaved_steps)
                model.load_state_dict(last_normal_state[0])  # 回退模型参数
                optimizer.load_state_dict(last_normal_state[1])  # 回退优化器参数
                epoch_train_acc.clear()
                epoch_train_loss.clear()
                print(f"Step {step}: loss=nan, 模型回退 {unsaved_steps} 步")
                step -= unsaved_steps  # 调整步骤计数
                continue

            flood.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            epoch_train_loss.append(loss.item())  # 累积训练损失
            epoch_train_acc.append((torch.argmax(outputs) == labels).sum().item() / labels.size(-1))  # 累积训练准确率
            progress_bar.update()  # 更新进度条

            # 验证阶段
            if step % val_per_step == 0:
                last_normal_state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

                model.eval()  # 切换到评估模式
                epoch_val_loss = []
                epoch_val_acc = []
                with torch.no_grad():  # 在验证阶段禁用梯度计算
                    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(random.randint(1 - 2**59, 2**64 - 1)))
                    for val_step, (inputs, labels) in enumerate(val_loader):
                        inputs, labels = inputs.to(device), labels.to(device).view(-1)
                        if val_step >= steps_to_val:
                            break  # 达到验证批次上限，停止验证

                        outputs = model(inputs).view(-1, VOCAB_SIZE)  # 前向传播
                        loss = F.cross_entropy(outputs, labels)  # 计算验证损失
                        epoch_val_loss.append(loss.item())  # 累计验证损失
                        epoch_val_acc.append((torch.argmax(outputs) == labels).sum().item() / labels.size(-1))  # 累积验证准确率
                        progress_bar.update()

                # 计算并记录损失、准确率的平均值和标准差
                train_loss_avg = sum(epoch_train_loss) / val_per_step
                train_loss_std = math.sqrt(sum((loss - train_loss_avg) ** 2 for loss in epoch_train_loss) / val_per_step)
                train_acc_avg = sum(epoch_train_acc) / len(epoch_train_acc)
                train_acc_std = math.sqrt(sum((acc - train_acc_avg) ** 2 for acc in epoch_train_acc) / len(epoch_train_acc))
                val_loss_avg = sum(epoch_val_loss) / steps_to_val
                val_loss_std = math.sqrt(sum((loss - val_loss_avg) ** 2 for loss in epoch_val_loss) / steps_to_val)
                val_acc_avg = sum(epoch_val_acc) / len(epoch_val_acc)
                val_acc_std = math.sqrt(sum((acc - val_acc_avg) ** 2 for acc in epoch_val_acc) / len(epoch_val_acc))

                train_loss.append(train_loss_avg)
                val_loss.append(val_loss_avg)
                train_accuracy.append(train_acc_avg)
                val_accuracy.append(val_acc_avg)

                print(
                    f"Validation Step {step // val_per_step}:",
                    f"train_loss={train_loss_avg:.3f},",
                    f"train_loss_std={train_loss_std:.3f},",
                    f"train_acc={train_acc_avg:.3f},",
                    f"train_acc_std={train_acc_std:.3f},",
                    f"val_loss={val_loss_avg:.3f},",
                    f"val_loss_std={val_loss_std:.3f},",
                    f"val_acc={val_acc_avg:.3f},",
                    f"val_acc_std={val_acc_std:.3f}"
                )

                # 将损失、准确率清空
                epoch_train_loss.clear()
                epoch_train_acc.clear()

                empty_cache()  # 清理缓存
                if step >= val_steps * val_per_step:
                    break
                model.train()  # 确保模型在训练模式下

    progress_bar.close()  # 关闭进度条

    return train_loss, val_loss, train_accuracy, val_accuracy, last_batch, generator_state


def plot_training_process(train_loss: list[float], val_loss: list[float], train_accuracy: list[float], val_accuracy: list[float], img_path: pathlib.Path | str):
    """
    绘制训练过程中的损失、准确率曲线。

    Args:
        train_loss: 两次验证之间的训练损失平均值。
        val_loss: 一次验证的验证损失平均值。
        train_accuracy: 两次验证之间的训练准确率平均值。
        val_accuracy: 一次验证的的验证准确率平均值。
        img_path: 图形保存的文件路径，可以是字符串或Path对象。
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))  # 创建一个图形和一组坐标轴

    # 绘制训练过程中的损失曲线
    ax1.plot(train_loss, label="Train Loss", color="red")

    # 绘制验证过程中的损失曲线
    val_steps = [x + 0.5 for x in range(len(train_loss))]
    ax1.plot(val_steps, val_loss, label="Validation Loss", color="blue")

    # 设置第一个Y轴的标签
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Validation Steps")

    # 创建第二个Y轴用于准确率
    ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴

    # 绘制训练过程中的准确率曲线
    ax2.plot(train_accuracy, label="Train Accuracy", color="green", linestyle="--")

    # 绘制验证过程中的准确率曲线
    ax2.plot(val_steps, val_accuracy, label="Validation Accuracy", color="blue", linestyle="--")

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
    """
    训练 MIDI 模型并绘制训练过程中的损失、准确率曲线。
    """
    # 用户定义的配置字典
    config = {
        "pretrained_ckpt": "/kaggle/input/tkaimidi/pytorch/default/1/ckpt",  # 预训练模型的检查点路径，可以为空
        "local_ckpt": "ckpt",  # 在本地保存的检查点的路径，训练结束后会保存到这里
        "external_datasets_path": ["/kaggle/input/music-midi"],  # 外部数据集的路径列表
        "lr": 1e-4,  # 学习率
        "weight_decay": 1e-5,  # 权重衰减系数
        "seq_size": 512,  # 输入序列的大小 - 1
        "val_steps": 12,  # 进行多少次验证步骤（因为 Dataset 进行了数据增强，一个 Epoch 训练数据变得很多，Kaggle GPU 12个小时跑不完，所以用验证步骤代替）
        "train_batch_size": 4,  # 训练时的批量大小
        "val_batch_size": 4,  # 验证时的批量大小，越大验证结果越准确，但是资源使用倍数增加，验证时间也增加（但没有资源使用增加得多）
        "train_length": 0.8,  # 训练集占数据集的比例，用来保证用来验证的数据不被训练
        "val_per_step": 4096,  # 每多少个训练步骤进行一次验证
        "steps_to_val": 256,  # 抽样验证，每一次验证使用多少个批次，越大验证结果越准确，但是验证时间倍数增加，资源使用不增加
        "flooding_level": 0  # 允许训练最低的损失值。训练损失低于该值则执行梯度上升，否则执行梯度下降
    }

    # 定义路径
    local_ckpt = pathlib.Path(config["local_ckpt"])
    local_dataset_path = pathlib.Path(tempfile.gettempdir())
    external_datasets_path = [pathlib.Path(path) for path in config["external_datasets_path"]]

    # 如果预训练检查点存在，则复制到本地检查点路径
    if config["pretrained_ckpt"] and pathlib.Path(config["pretrained_ckpt"]).exists():
        if local_ckpt.exists():
            shutil.rmtree(local_ckpt)  # 删除现有的本地检查点
        shutil.copytree(config["pretrained_ckpt"], local_ckpt, dirs_exist_ok=True)  # 复制预训练检查点到本地

    # 复制外部数据集到本地
    for path in external_datasets_path:
        shutil.copytree(path, local_dataset_path / path.name, dirs_exist_ok=True)

    # 加载训练数据集
    dataset = MidiDataset(local_dataset_path, seq_size=config["seq_size"])

    # 获取设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = MidiNet()

    # 创建优化器
    create_optimizer = partial(optim.Adam, model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # 尝试加载检查点
    try:
        model_state, optimizer_state, old_train_loss, old_val_loss, old_train_accuracy, old_val_accuracy, dataset_length, train_start, last_batch, generator_state = load_checkpoint(local_ckpt, train=True)
        model.load_state_dict(model_state)  # 加载模型状态
        model = model.to(device)  # 转移到指定设备
        optimizer = create_optimizer()  # 初始化优化器
        optimizer.load_state_dict(optimizer_state)

        # 检查数据集大小是否匹配
        if dataset_length != len(dataset):
            print(f"数据集大小不匹配，可能是使用了与之前不同的数据集训练: {dataset_length} 与 {len(dataset)} 不匹配", file=sys.stderr)
            train_start = random.randint(0, len(dataset) - 1)  # 随机选择新的训练起点
            generator_state = ...  # 重置随机数生成器状态
            last_batch = 0  # 重置批次计数
    except Exception as e:
        print(f"加载检查点时发生错误: {e}", file=sys.stderr)
        model = model.to(device)  # 转移到指定设备
        optimizer = create_optimizer()  # 初始化优化器
        old_train_loss, old_val_loss, old_train_accuracy, old_val_accuracy = [], [], [], []  # 初始化损失和准确率记录
        train_start = random.randint(0, len(dataset) - 1)  # 随机选择训练起点
        generator_state = ...  # 初始化随机数生成器状态
        last_batch = 0  # 初始化批次计数

    # 开始训练模型
    train_loss, val_loss, train_accuracy, val_accuracy, last_batch, generator_state = train(
        model, dataset, optimizer,
        val_steps=config["val_steps"],
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        train_length=config["train_length"],
        val_per_step=config["val_per_step"],
        steps_to_val=config["steps_to_val"],
        flooding_level=config["flooding_level"],
        train_start=train_start,
        generator_state=generator_state,
        last_batch=last_batch,
        device=device
    )

    # 合并旧的损失记录和新的记录
    train_loss = old_train_loss + train_loss
    val_loss = old_val_loss + val_loss
    train_accuracy = old_train_accuracy + train_accuracy
    val_accuracy = old_val_accuracy + val_accuracy

    # 保存当前模型的检查点
    save_checkpoint(
        model, optimizer, train_loss, val_loss, train_accuracy, val_accuracy,
        len(dataset), train_start, last_batch, generator_state, local_ckpt
    )

    # 绘制训练过程中的损失和准确率曲线
    plot_training_process(train_loss, val_loss, train_accuracy, val_accuracy, "statistics.png")


if __name__ == "__main__":
    main()
