# 训练的代码
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
    from model import MidiNet, TIME_PRECISION, MAX_TIME_DIFF, NOTE_DURATION_COUNT, load_checkpoint, save_checkpoint
    from utils import midi_to_notes, norm_data, empty_cache


class MidiDataset(Dataset):
    """
    MIDI 数据集，继承自 PyTorch 的 Dataset 类，允许加载和处理 MIDI 文件数据。

    Args:
        path: MIDI 文件所在的路径
        seq_size: 每个序列的长度
    """

    def __init__(self, path: pathlib.Path, seq_size: int):
        self.data: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []  # 存储每个 MIDI 文件的数据
        self.seq_size = seq_size  # 每个序列的长度
        self.length = 0  # 数据集的总长度

        # 获取所有 MIDI 文件路径
        midi_files = sorted(list(path.glob("**/*.mid")), key=lambda x: x.name)
        progress_bar = tqdm.tqdm(desc="Load Dataset", total=len(midi_files))

        for filepath in midi_files:
            # 解析 MIDI 文件，提取音符信息
            parsed_data = [(note, start_at) for _, note, start_at, _ in midi_to_notes(mido.MidiFile(filepath, clip=True))]

            for param_strict in [False, True]:
                data = norm_data(parsed_data, TIME_PRECISION, MAX_TIME_DIFF, strict=param_strict)  # 规范化数据

                # 处理音符数过少的情况，重复音符数据填充
                if len(data) < seq_size:
                    data[0] = (data[0][0], NOTE_DURATION_COUNT - 1)  # 修正第一个音符的时间
                    new_data = data * (seq_size // len(data) + 1)  # 重复数据，确保序列长度足够
                    new_data[0] = (new_data[0][0], 0)  # 还原第一个音符的时间
                    data = new_data

                # 计算每个序列的可扩展长度
                offsets: list[tuple[int, int]] = []
                for i in range(len(data) - seq_size):
                    seq_notes, seq_times = zip(*data[i:i + seq_size])
                    note_offsets = 128 - max(seq_notes)
                    time_offsets = (NOTE_DURATION_COUNT - 1) // (max(seq_times) // math.gcd(*seq_times))
                    offsets.append((note_offsets, time_offsets))
                    self.length += note_offsets * time_offsets

                # 将处理过的数据和偏移量存储到数据集中
                self.data.append((offsets, data))
            progress_bar.update()

        progress_bar.close()

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        for offsets, midi_notes_data in self.data:
            found = None
            # 遍历每个 MIDI 文件中的序列，找到对应的索引
            for seq_index, (note_offsets, time_offsets) in enumerate(offsets):
                seq_offsets = note_offsets * time_offsets
                if index >= seq_offsets:
                    index -= seq_offsets  # 如果当前序列的音符不足，则调整索引
                else:
                    found = seq_index
                    time_offset, note_offset = divmod(index, note_offsets)  # 获取当前序列的偏移量
                    break

            if found is not None:
                # 提取序列，并应用偏移量进行数据增强
                seq_notes, seq_times = zip(*midi_notes_data[found:found + self.seq_size])
                time_gcd = math.gcd(*seq_times)
                seq = [(note + note_offset) * NOTE_DURATION_COUNT + (time // time_gcd * (time_offset + 1)) for note, time in zip(seq_notes, seq_times)]

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


def train(model: MidiNet, dataset: MidiDataset, optimizer: optim.SGD, train_batch_size: int, val_batch_size: int, train_length: float = 0.8, val_steps: int = 256, val_per_step: int = 4096, steps_to_val: int = 256, train_start: int = 0, generator_state: torch.tensor = ..., last_batch: int = 0, device: torch.device = torch.device("cpu")) -> tuple[list[float], list[float], list[float], list[float], int, torch.tensor]:
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

    step = train_loss_sum = train_accuracy_sum = 0
    model.train()  # 确保模型在训练模式下
    train_begin = True  # 刚开始训练
    while step < val_steps * val_per_step:  # 训练直到达到验证步骤上限
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
            outputs = model(inputs).view(-1, NOTE_DURATION_COUNT * 128)  # 前向传播

            # 计算损失并进行反向传播
            loss = F.cross_entropy(outputs, labels)  # 计算交叉熵损失

            # 检查损失是否为 NaN
            if torch.isnan(loss):
                unsaved_steps = step % val_per_step
                progress_bar.update(1 - unsaved_steps)
                model.load_state_dict(last_normal_state[0])  # 回退模型参数
                optimizer.load_state_dict(last_normal_state[1])  # 回退优化器参数
                train_loss_sum = train_accuracy_sum = 0
                print(f"Step {step}: loss=nan, 模型回退 {unsaved_steps} 步")
                step -= unsaved_steps  # 调整步骤计数
                continue

            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            train_loss_sum += loss.item()  # 累积训练损失
            train_accuracy_sum += (torch.argmax(outputs) == labels).sum().item()  # 累积训练准确率
            progress_bar.update()  # 更新进度条

            # 验证阶段
            if step % val_per_step == 0:
                last_normal_state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

                model.eval()  # 切换到评估模式
                val_loss_sum = val_accuracy_sum = 0
                with torch.no_grad():  # 在验证阶段禁用梯度计算
                    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(random.randint(1 - 2**59, 2**64 - 1)))
                    for val_step, (inputs, labels) in enumerate(val_loader):
                        inputs, labels = inputs.to(device), labels.to(device).view(-1)
                        if val_step >= steps_to_val:
                            break  # 达到验证批次上限，停止验证

                        outputs = model(inputs).view(-1, NOTE_DURATION_COUNT * 128)  # 前向传播
                        loss = F.cross_entropy(outputs, labels)  # 计算验证损失
                        val_loss_sum += loss.item()  # 累计验证损失
                        val_accuracy_sum += (torch.argmax(outputs) == labels).sum().item()  # 累积验证准确率
                        progress_bar.update()

                # 计算并记录平均损失
                train_loss_avg = train_loss_sum / val_per_step
                val_loss_avg = val_loss_sum / steps_to_val
                train_acc_avg = train_accuracy_sum / val_per_step / labels.size(0)
                val_acc_avg = val_accuracy_sum / steps_to_val / labels.size(0)

                train_loss.append(train_loss_avg)
                val_loss.append(val_loss_avg)
                train_accuracy.append(train_acc_avg)
                val_accuracy.append(val_acc_avg)

                print(f"Validation Step {step // val_per_step}:", f"train_loss={train_loss_avg:.3f},", f"val_loss={val_loss_avg:.3f},", f"train_acc={train_acc_avg:.3f},", f"val_acc={val_acc_avg:.3f}")
                train_loss_sum = train_accuracy_sum = 0  # 将训练损失、准确率计数器归零

                empty_cache()  # 清理缓存
                if step >= val_steps * val_per_step:
                    break
                model.train()  # 确保模型在训练模式下

    progress_bar.close()  # 关闭进度条

    return train_loss, val_loss, train_accuracy, val_accuracy, last_batch, train_generator.get_state()


def plot_training_process(train_loss: list[float], val_loss: list[float], train_accuracy: list[float], val_accuracy: list[float], img_path: pathlib.Path | str):
    """
    绘制训练过程中的损失曲线，包括训练集和验证集的音符损失与时间损失。

    Args:
        train_loss: 每一验证步数过程中的训练损失值。
        val_loss: 每一验证步数过程中的验证损失值。
        train_accuracy: 每一验证步数过程中的训练准确率值。
        val_accuracy: 每一验证步数过程中的验证准确率值。
        img_path: 图形保存的文件路径，可以是字符串或Path对象。
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))  # 创建一个图形和一组坐标轴

    # 绘制训练过程中的损失曲线
    ax1.plot(train_loss, label="Train Loss", color="red")

    # 绘制验证过程中的损失曲线
    val_steps = range(1, len(train_loss) + 1)
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
    ax1.set_title("Training Process")
    ax1.legend(loc="upper left")  # 为损失曲线添加图例，并设置位置
    ax2.legend(loc="upper right")  # 为准确率曲线添加图例，并设置位置

    plt.tight_layout()  # 自动调整布局，防止标签重叠
    pathlib.Path(img_path).parent.mkdir(parents=True, exist_ok=True)  # 确保保存路径存在
    plt.savefig(img_path, dpi=300, bbox_inches="tight")  # 保存图形
    plt.show()  # 显示图形


def main():
    """
    训练 MIDI 模型并绘制训练过程中的损失曲线。
    """
    # 用户定义的配置字典
    config = {
        "pretrained_ckpt": "/kaggle/input/tkaimidi/pytorch/default/1/ckpt",  # 预训练模型的检查点路径，可以为空
        "local_ckpt": "ckpt",  # 在本地保存的检查点的路径，训练结束后会保存到这里
        "external_datasets_path": ["/kaggle/input/music-midi"],  # 外部数据集的路径列表
        "lr": 0.003548133892335713,  # 学习率
        "weight_decay": 1e-3,  # 权重衰减系数
        "seq_size": 512,  # 输入序列的大小 - 1
        "val_steps": 1,  # 进行多少次验证步骤（因为 Dataset 进行了数据增强，一个 Epoch 训练数据变得很多，Kaggle GPU 12个小时跑不完，所以用验证步骤代替）
        "train_batch_size": 1,  # 训练时的批量大小
        "val_batch_size": 4,  # 验证时的批量大小，越大验证结果越准确，但是资源使用倍数增加，验证时间也增加（但没有资源使用增加得多）
        "train_length": 0.8,  # 训练集占数据集的比例，用来保证用来验证的数据不被训练
        "val_per_step": 4096,  # 每多少个训练步骤进行一次验证
        "steps_to_val": 256  # 抽样验证，每一次验证使用多少个批次，越大验证结果越准确，但是验证时间倍数增加，资源使用不增加
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

    dataset = MidiDataset(local_dataset_path, seq_size=config["seq_size"])  # 加载训练数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取设备
    model = MidiNet()  # 初始化模型

    # 尝试加载检查点
    create_optimizer = partial(optim.SGD, model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"])

    try:
        model_state, optimizer_state, old_train_loss, old_val_loss, old_train_accuracy, old_val_accuracy, dataset_length, train_start, last_batch, generator_state = load_checkpoint(local_ckpt, train=True)
        model.load_state_dict(model_state)  # 加载模型状态
        model = model.to(device)  # 转移到指定设备并编译模型
        optimizer = create_optimizer()  # 初始化优化器
        optimizer.load_state_dict(optimizer_state)

        if dataset_length != len(dataset):
            print(f"数据集大小不匹配，可能是使用了与之前不同的数据集训练: {dataset_length} 与 {len(dataset)} 不匹配", file=sys.stderr)
            train_start = random.randint(0, len(dataset) - 1)
            generator_state = ...
            last_batch = 0
    except Exception as e:
        print(f"加载检查点时发生错误: {e}", file=sys.stderr)
        optimizer = create_optimizer()  # 初始化优化器
        old_train_loss, old_val_loss, old_train_accuracy, old_val_accuracy = [], [], [], []
        train_start = random.randint(0, len(dataset) - 1)
        generator_state = ...
        last_batch = 0

    # 开始训练模型
    train_loss, val_loss, train_accuracy, val_accuracy, last_batch, generator_state = train(model, dataset, optimizer, val_steps=config["val_steps"], train_batch_size=config["train_batch_size"], val_batch_size=config["val_batch_size"], train_length=config["train_length"], val_per_step=config["val_per_step"], steps_to_val=config["steps_to_val"], train_start=train_start, generator_state=generator_state, last_batch=last_batch, device=device)

    # 合并旧的损失记录和新的记录
    train_loss = old_train_loss + train_loss
    val_loss = old_val_loss + val_loss
    train_accuracy = old_train_accuracy + train_accuracy
    val_accuracy = old_val_accuracy + val_accuracy

    save_checkpoint(model, optimizer, train_loss, val_loss, train_accuracy, val_accuracy, len(dataset), train_start, last_batch, generator_state, local_ckpt)  # 保存当前模型的检查点
    plot_training_process(train_loss, val_loss, train_accuracy, val_accuracy, "statistics.png")  # 绘制变化曲线


if __name__ == "__main__":
    main()
