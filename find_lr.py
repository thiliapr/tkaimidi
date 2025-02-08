# 寻找最佳学习率
# Copyright (C) thiliapr 2025
# License: AGPLv3-or-later

import math
import pathlib
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from scipy.signal import savgol_filter

# 在非Jupyter环境下导入模型和工具库
if "get_ipython" not in globals():
    from model import MidiNet, NOTE_DURATION_COUNT
    from train import MidiDataset
    display = print
else:
    from IPython.display import display


def main():
    # 初始化学习率相关参数
    init_lr = 1e-6  # 初始学习率
    final_lr = 1e-1  # 最终学习率
    lr_precision = 10 ** (1 / 100)  # 学习率调整的步长比例

    # 定义文件路径
    local_dataset_path = pathlib.Path("/kaggle/input/music-midi")  # 数据集路径

    # 加载数据集和初始化模型
    dataset = MidiDataset(local_dataset_path, seq_size=512)  # 创建数据集实例
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 初始化加速器
    model = MidiNet()  # 初始化模型
    model = model.to(device)  # 将模型移动到设备
    optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-3)  # 初始化优化器

    # 准备数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)  # 创建数据加载器

    # 检查是否使用多GPU
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # 使用 DataParallel 进行多GPU训练

    # 设置初始学习率
    optimizer.param_groups[0]["lr"] = init_lr

    # 记录学习率和损失
    lr_loss_list = []

    # 使用tqdm来显示进度条
    dataloader_iter = iter(dataloader)
    for i in tqdm.tqdm(range(math.ceil(math.log(final_lr / init_lr, lr_precision)))):
        inputs, labels = next(dataloader_iter)
        inputs, labels = inputs.to(device), labels.to(device).view(-1)  # 调整标签形状
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs).view(-1, NOTE_DURATION_COUNT * 128)  # 前向传播，调整输出形状
        loss = F.cross_entropy(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        # 计算并记录学习率、损失和损失变化
        if i != 0:
            lr_loss_list.append((optimizer.param_groups[0]["lr"], loss.item(), lr_loss_list[-1][1] - loss.item()))
        else:
            lr_loss_list.append((optimizer.param_groups[0]["lr"], loss.item(), 0))

        # 调整学习率
        optimizer.param_groups[0]["lr"] *= lr_precision

    # 过滤掉包含NaN损失的记录
    result = list(filter(lambda x: not math.isnan(x[1]), lr_loss_list))
    result = [(index, lr, loss, diff) for index, (lr, loss, diff) in zip(range(len(result)), lr_loss_list)]
    indexes, lrs, losses, differences = zip(*result)

    # 保存结果为表格
    pd.DataFrame({"index": indexes, "lr": lrs, "loss": losses, "diff": differences}).sort_values(by="diff", ascending=False).to_csv("lr-loss-diff.csv", index=False)

    # 平滑处理并绘制损失曲线
    y_smooth = savgol_filter(losses, window_length=min(11, len(losses) - 1), polyorder=3)
    plt.plot(indexes, y_smooth)
    plt.title("Smoothed Loss Curve")
    plt.xlabel(f"Learning Rate (lr = {init_lr} * ({lr_precision} ** x))")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("LR-Loss Curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 将损失变化最大的前 10 位转化为 DataFrame 并显示
    sorted_result = sorted(result, key=lambda x: x[3], reverse=True)
    indexes, lrs, losses, differences = zip(*sorted_result[:10])

    df = pd.DataFrame({"index": indexes, "lr": lrs, "loss": losses, "diff": differences})
    display(df)


if __name__ == "__main__":
    main()
