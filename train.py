# MIDI 音乐生成模型训练模块
"""
MIDI 音乐生成模型训练模块

本模块提供完整的MIDI音乐生成模型训练流程，包含数据集处理、模型训练、验证评估及可视化功能

特性:
- 容错机制: 自动回退异常训练状态
- Flooding Loss: 训练损失低于某值时执行梯度上升，有利于提高泛化性能。论文见: https://arxiv.org/abs/2002.08709

使用示例:
>>> from train import main
>>> main()  # 启动完整训练流程

配置参数说明见main()函数中的config字典
"""

# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

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
    from model import MidiNet, TIME_PRECISION, VOCAB_SIZE, load_checkpoint, save_checkpoint
    from utils import midi_to_notes, normalize_times, notes_to_sheet, sheet_to_model, empty_cache


class MidiDataset(Dataset):
    """
    MIDI 数据集类，用于加载和处理MIDI文件生成训练序列

    特性:
    - 自动处理不同长度的MIDI文件
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
        midi_files = sorted(list(path.glob("**/*.mid")), key=lambda x: x.name)
        # midi_files = list(filter(lambda x: x.name.startswith("Touhou"), path.glob("**/*.mid")))  # 测试用

        if show_progress:
            progress_bar = tqdm.tqdm(desc="Load Dataset", total=len(midi_files))

        for filepath in midi_files:
            # 解析 MIDI 文件，提取音符信息
            parsed_data = [(note, start_at) for _, note, start_at, _ in midi_to_notes(mido.MidiFile(filepath, clip=True))]

            # 规范化时间
            normalized_data = normalize_times(parsed_data, TIME_PRECISION, strict=True)

            # 转换格式
            data = sheet_to_model(notes_to_sheet(normalized_data, 128))

            # 通过循环填充达到最小长度要求
            if len(data) < seq_size:
                normalized_data[0] = (normalized_data[0][0], interval_repeat)
                normalized_data *= seq_size // len(data) + 1  # 重复数据，确保序列长度足够
                normalized_data[0] = (normalized_data[0][0], 0)
                data = sheet_to_model(notes_to_sheet(normalized_data, 128))

            # 探测每个子序列的位置
            seq_indexes: list[int] = []
            after_interval = True  # 指针在至少一个时间间隔之后
            allow_note = True  # 是否允许音符
            for i in range(len(data) - seq_size):
                cont = False
                if 36 <= data[i] <= 39 and after_interval and allow_note:
                    allow_note = False
                    cont = True
                elif data[i] == 40:
                    after_interval = True
                elif data[i] < 12:
                    if after_interval and allow_note:
                        cont = True
                    allow_note = True
                    after_interval = False
                if not cont:
                    continue

                seq_indexes.append(i)

            self.length += len(seq_indexes)
            self.data.append((data, seq_indexes))

            if show_progress:
                progress_bar.update()
        if show_progress:
            progress_bar.close()

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        for filedata, seq_indexes in self.data:
            if index >= len(seq_indexes):
                index -= len(seq_indexes)
                continue
            seq_index = seq_indexes[index]
            seq = filedata[seq_index:seq_index + self.seq_size]

        # 返回输入和目标序列
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


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
    num_epochs: int = 256,
    train_batch_size: int = 1,
    val_batch_size: int = 2,
    train_length: float = 0.8,
    steps_to_val: int = 256,
    flooding_level: float = 0,
    train_start: int = 0,
    device: torch.device = torch.device("cpu")
) -> tuple[list[list[float]], list[float], list[float], list[float]]:
    """
    训练模型并记录训练和验证的损失。

    Args:
        model: 要训练的模型
        dataset: 用于训练的 MIDI 数据集
        optimizer: 优化器，用于更新模型参数
        num_epochs: 训练多少个Epoch
        train_batch_size: 每个训练批次的样本数
        val_batch_size: 每个验证批次的样本数
        train_length: 训练集占数据集的比例
        steps_to_val: 每一次验证使用多少个批次
        flooding_level: 允许训练最低的损失值。训练损失低于该值则执行梯度上升，否则执行梯度下降
        train_start: 拆分数据集时训练集的开始索引
        device: 用于训练的设备

    Returns:
        训练、验证损失和准确率的历史记录
    """
    empty_cache()  # 清理缓存以释放内存

    # 根据 train_length 切分训练集和验证集
    train_dataset, val_dataset = split_dataset(dataset, train_length, train_start)
    steps_to_val = min(steps_to_val, len(val_dataset) // val_batch_size)  # 确保验证步骤不超过验证集大小

    # 获取训练数据加载器
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

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
    progress_bar = tqdm.tqdm(desc="Training", total=num_epochs * (len(train_loader) + steps_to_val))

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()  # 确保模型在训练模式下
        epoch_train_loss, epoch_train_acc = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            optimizer.zero_grad()  # 清空优化器中的梯度
            outputs = model(inputs).view(-1, VOCAB_SIZE)  # 前向传播
            loss = F.cross_entropy(outputs, labels)  # 计算交叉熵损失

            # 检查损失是否为 NaN
            if torch.isnan(loss):
                model.load_state_dict(last_normal_state[0])  # 回退模型参数
                optimizer.load_state_dict(last_normal_state[1])  # 回退优化器参数
                epoch_train_acc.clear()
                epoch_train_loss.clear()
                continue

            flood = (loss - flooding_level).abs() + flooding_level  # 梯度下降或上升
            flood.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            epoch_train_loss.append(loss.item())  # 累积训练损失
            epoch_train_acc.append((torch.argmax(outputs) == labels).sum().item() / labels.size(-1))  # 累积训练准确率
            progress_bar.update()  # 更新进度条

        # 验证阶段
        last_normal_state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

        model.eval()  # 切换到评估模式
        epoch_val_loss, epoch_val_acc = [], []
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
        train_loss_avg = sum(epoch_train_loss) / len(epoch_train_loss)
        train_acc_avg = sum(epoch_train_acc) / len(epoch_train_acc)
        train_acc_std = math.sqrt(sum((acc - train_acc_avg) ** 2 for acc in epoch_train_acc) / len(epoch_train_acc))
        val_loss_avg = sum(epoch_val_loss) / steps_to_val
        val_loss_std = math.sqrt(sum((loss - val_loss_avg) ** 2 for loss in epoch_val_loss) / steps_to_val)
        val_acc_avg = sum(epoch_val_acc) / len(epoch_val_acc)
        val_acc_std = math.sqrt(sum((acc - val_acc_avg) ** 2 for acc in epoch_val_acc) / len(epoch_val_acc))

        train_loss.append(epoch_train_loss)
        val_loss.append(val_loss_avg)
        train_accuracy.append(train_acc_avg)
        val_accuracy.append(val_acc_avg)

        print(
            f"Epoch {epoch + 1}:",
            f"train_loss={train_loss_avg:.3f},",
            f"train_acc={train_acc_avg:.3f},",
            f"train_acc_std={train_acc_std:.3f},",
            f"val_loss={val_loss_avg:.3f},",
            f"val_loss_std={val_loss_std:.3f},",
            f"val_acc={val_acc_avg:.3f},",
            f"val_acc_std={val_acc_std:.3f}"
        )

    progress_bar.close()  # 关闭进度条

    return train_loss, val_loss, train_accuracy, val_accuracy


def plot_training_process(train_loss: list[list[float]], val_loss: list[float], train_accuracy: list[float], val_accuracy: list[float], img_path: pathlib.Path | str):
    """
    绘制训练过程中的损失、准确率曲线。

    Args:
        train_loss: 每个Epoch的每一步的训练损失值。
        val_loss: 每个Epoch的验证损失平均值。
        train_accuracy: 每个Epoch的训练准确率平均值。
        val_accuracy: 每个Epoch的的验证准确率平均值。
        img_path: 图形保存的文件路径，可以是字符串或Path对象。
    """
    def smooth(losses: list[float], max_diff: float = 0.1):
        mean = sum(losses) / len(losses)
        last, next = mean, losses[1]

        smoothed = losses.copy()
        for i, loss in enumerate(losses):
            if max(abs(loss - last), abs(loss - next)) > max_diff:
                smoothed[i] = (last * 3 + next + mean * 28) / 32
            last = (last * 0.9 + smoothed[i]) / 1.9
            next = smoothed[i + 2] if i < len(smoothed) - 2 else mean
        return smoothed

    fig, ax1 = plt.subplots(figsize=(10, 6))  # 创建一个图形和一组坐标轴

    # 绘制训练过程中的损失曲线
    train_loss_x = [
        epoch + epoch_step / len(epoch_loss)
        for epoch, epoch_loss in enumerate(train_loss)
        for epoch_step in range(len(epoch_loss))
    ]
    train_loss_y = smooth([loss for epoch in train_loss for loss in epoch])
    ax1.plot(train_loss_x, train_loss_y, label="Train Loss", color="red")

    # 绘制验证过程中的损失曲线
    val_steps = list(range(1, len(train_loss) + 1))
    ax1.plot(val_steps, val_loss, label="Validation Loss", color="blue")

    # 设置第一个Y轴的标签
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epochs")

    # 创建第二个Y轴用于准确率
    ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴

    # 绘制训练过程中的准确率曲线
    ax2.plot(val_steps, train_accuracy, label="Train Accuracy", color="green", linestyle="--")

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
        "lr": 1e-3,  # 学习率
        "weight_decay": 1e-2,  # 权重衰减系数
        "seq_size": 768,  # 输入序列的大小 - 1
        "num_epochs": 1,  # 训练多少个Epoch
        "train_batch_size": 12,  # 训练时的批量大小
        "val_batch_size": 16,  # 验证时的批量大小，越大验证结果越准确，但是资源使用倍数增加，验证时间也增加（但没有资源使用增加得多）
        "train_length": 0.8,  # 训练集占数据集的比例，用来保证用来验证的数据不被训练
        "steps_to_val": 128,  # 抽样验证，每一次验证使用多少个批次，越大验证结果越准确，但是验证时间倍数增加，资源使用不增加
        "flooding_level": 0  # 允许训练最低的损失值。训练损失低于该值则执行梯度上升，否则执行梯度下降
    }

    # 定义路径
    local_ckpt = pathlib.Path(config["local_ckpt"])
    local_dataset_path = pathlib.Path(tempfile.gettempdir()) / random.randbytes(8).hex()[2:]
    external_datasets_path = [pathlib.Path(path) for path in config["external_datasets_path"]]

    # 如果预训练检查点存在，则复制到本地检查点路径
    if config["pretrained_ckpt"] and pathlib.Path(config["pretrained_ckpt"]).exists():
        if local_ckpt.exists():
            shutil.rmtree(local_ckpt)  # 删除现有的本地检查点
        shutil.copytree(config["pretrained_ckpt"], local_ckpt, dirs_exist_ok=True)  # 复制预训练检查点到本地

    # 复制外部数据集到本地
    local_dataset_path.mkdir(exist_ok=True)
    for path in external_datasets_path:
        shutil.copytree(path, local_dataset_path / path.name, dirs_exist_ok=True)

    # 加载训练数据集
    dataset = MidiDataset(local_dataset_path, seq_size=config["seq_size"])

    # 获取设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = MidiNet()

    # 创建优化器
    create_optimizer = partial(optim.AdamW, model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # 尝试加载检查点
    try:
        model_state, optimizer_state, old_train_loss, old_val_loss, old_train_accuracy, old_val_accuracy, dataset_length, train_start = load_checkpoint(local_ckpt, train=True)
        if model_state:  # 如果模型状态不为空，则尝试加载模型状态
            model.load_state_dict(model_state)
        model = model.to(device)  # 转移到指定设备

        optimizer = create_optimizer()  # 初始化优化器
        if optimizer_state:  # 如果优化器状态不为空，则尝试加载模型状态
            optimizer.load_state_dict(optimizer_state)

        # 检查数据集大小是否匹配
        if dataset_length == -1:
            dataset_length = "N/A"
        if dataset_length != len(dataset):
            print(f"数据集大小不匹配，可能是使用了与之前不同的数据集训练: {dataset_length} 与 {len(dataset)} 不匹配", file=sys.stderr)
            train_start = random.randint(0, len(dataset) - 1)  # 随机选择新的训练起点
    except Exception as e:
        print(f"加载检查点时发生错误: {e}", file=sys.stderr)
        model = model.to(device)  # 转移到指定设备
        optimizer = create_optimizer()  # 初始化优化器
        old_train_loss, old_val_loss, old_train_accuracy, old_val_accuracy = [], [], [], []  # 初始化损失和准确率记录
        train_start = random.randint(0, len(dataset) - 1)  # 随机选择训练起点

    # 开始训练模型
    train_loss, val_loss, train_accuracy, val_accuracy = train(
        model, dataset, optimizer,
        num_epochs=config["num_epochs"],
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        train_length=config["train_length"],
        steps_to_val=config["steps_to_val"],
        flooding_level=config["flooding_level"],
        train_start=train_start,
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
        len(dataset), train_start, local_ckpt
    )

    # 绘制训练过程中的损失和准确率曲线
    plot_training_process(train_loss, val_loss, train_accuracy, val_accuracy, "statistics.png")

    # 删除音乐文件临时储存
    shutil.rmtree(local_dataset_path)


if __name__ == "__main__":
    main()
