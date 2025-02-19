"""
学习率探索工具: 基于损失曲线变化寻找最佳学习率范围

实现方法:
1. 采用指数增长学习率策略逐步增大学习率
2. 监控损失函数值及其变化率
3. 使用Savitzky-Golay滤波器平滑损失曲线
4. 通过损失变化率确定最佳学习率区间

参考: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
"""

# Copyright (C) thiliapr 2025
# License: AGPLv3-or-later

import math
import pathlib
import itertools

import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from scipy.signal import savgol_filter

# 常量定义
INIT_LR = 1e-6  # 初始学习率 (建议范围: 1e-8 ~ 1e-5)
FINAL_LR = 1e-1  # 终止学习率 (建议范围: 1e-2 ~ 1.0)
LR_GROWTH_FACTOR = 1.02  # 学习率增长因子 (建议范围: 1.01 ~ 1.05)
BATCH_SIZE = 4  # 批处理大小 (根据显存调整)
SEQ_LENGTH = 512  # 序列长度 (需与训练参数一致)
DATASET_PATH = pathlib.Path("/kaggle/input/music-midi")

# 在非Jupyter环境下导入模型和工具库
if "get_ipython" not in globals():
    from model import MidiNet, NOTE_DURATION_COUNT, MAX_NOTE
    from train import MidiDataset
    display = print
else:
    from IPython.display import display


def setup_environment():
    """
    初始化训练环境

    Returns:
        设备对象和数据加载器

    Raises:
        当数据集路径不存在时抛出
    """
    # 检查数据集路径
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"数据集路径不存在: {DATASET_PATH}")

    # 设备初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载配置
    dataset = MidiDataset(DATASET_PATH, seq_size=SEQ_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    return device, dataloader


def initialize_model(device: torch.device) -> torch.nn.Module:
    """
    初始化模型并配置并行训练

    Args:
        device: 训练设备

    Returns:
        配置好的模型实例
    """
    model = MidiNet().to(device)

    # 多GPU并行配置
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} GPUs 进行并行训练")
        model = DataParallel(model)

    return model


def lr_exploration_loop(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device
) -> list[tuple[float, float, float]]:
    """
    执行学习率探索循环

    Args:
        model: 待训练模型
        optimizer: 优化器
        dataloader: 数据加载器
        device: 训练设备

    Returns:
        学习率记录 (学习率，损失值，损失变化率)
    """
    learning_records = []
    previous_loss = None

    # 创建循环迭代器避免数据耗尽
    dataloader_iter = itertools.cycle(dataloader)

    # 计算总迭代次数
    total_steps = math.ceil(math.log(FINAL_LR / INIT_LR, LR_GROWTH_FACTOR))

    for _ in tqdm.tqdm(range(total_steps), desc="Test LR"):
        # 获取数据批次
        inputs, labels = next(dataloader_iter)
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs).view(-1, NOTE_DURATION_COUNT * (MAX_NOTE + 1))
        loss = F.cross_entropy(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录训练指标
        current_lr = optimizer.param_groups[0]["lr"]
        loss_value = loss.item()

        # 计算损失变化率
        loss_diff = previous_loss - loss_value if previous_loss is not None else 0
        learning_records.append((current_lr, loss_value, loss_diff))
        previous_loss = loss_value

        # 更新学习率 (指数增长)
        optimizer.param_groups[0]["lr"] *= LR_GROWTH_FACTOR

    return learning_records


def analyze_results(records: list[tuple[float, float, float]]):
    """
    分析并可视化结果

    Args:
        records: 学习率探索记录
    """
    # 数据清洗
    clean_records = [x for x in records if not math.isnan(x[1])]

    # 转换为DataFrame
    df = pd.DataFrame(clean_records, columns=["lr", "loss", "loss_diff"])
    df["log_lr"] = df["lr"].apply(math.log10)

    # 保存原始数据
    df.to_csv("lr_exploration.csv", index=False)

    # 损失曲线平滑
    window_size = min(21, len(df) // 2)  # 自适应窗口大小
    if window_size % 2 == 0:
        window_size -= 1  # 确保窗口大小为奇数

    df["smooth_loss"] = savgol_filter(df["loss"], window_size, 3)

    # 绘制双对数曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df["log_lr"], df["smooth_loss"], linewidth=2)
    plt.title("LR & Loss", fontsize=14)
    plt.xlabel("log10(Learning Rate)", fontsize=12)
    plt.ylabel("Smoothed Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lr_exploration_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 显示最佳候选学习率
    df_top = df.nsmallest(10, "smooth_loss")
    display(df_top[["lr", "loss", "smooth_loss"]].style.format({
        "lr": "{:.2e}",
        "loss": "{:.4f}",
        "smooth_loss": "{:.4f}"
    }))


def main():
    """主执行流程"""
    try:
        # 环境初始化
        device, dataloader = setup_environment()

        # 模型初始化
        model = initialize_model(device)

        # 优化器配置
        optimizer = optim.SGD(
            model.parameters(),
            lr=INIT_LR,
            momentum=0.9,
            weight_decay=1e-3,
            nesterov=True
        )

        # 执行学习率探索
        records = lr_exploration_loop(model, optimizer, dataloader, device)

        # 结果分析
        analyze_results(records)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()
