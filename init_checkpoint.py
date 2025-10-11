# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import shutil
import random
import pathlib
import argparse
from typing import Optional
import torch
import numpy as np
from torch import optim
from utils.checkpoint import save_checkpoint
from utils.constants import DEFAULT_DIM_FEEDFORWARD, DEFAULT_DIM_HEAD, DEFAULT_NUM_DECODER_LAYERS, DEFAULT_NUM_ENCODER_LAYERS, DEFAULT_NUM_HEADS, DEFAULT_NUM_PITCH_LAYERS, DEFAULT_PITCH_CONV1_KERNEL, DEFAULT_PITCH_CONV2_KERNEL, DEFAULT_PITCH_DIM_FEEDFORWARD, DEFAULT_PITCH_DIM_MODEL, DEFAULT_VARIANCE_BINS, DEFAULT_NUM_NOTE_COUNT_LAYERS, DEFAULT_NUM_PITCH_MEAN_LAYERS, DEFAULT_NUM_PITCH_RANGE_LAYERS
from utils.model import MidiNet, MidiNetConfig


def set_seed(seed: int):
    """
    设置所有随机源的种子以确保实验可复现性。

    工作流程:
    1. 设置 Python 内置 random 模块的种子
    2. 设置 NumPy 的随机种子
    3. 设置 PyTorch 的 CPU 和 GPU 随机种子
    4. 配置 CuDNN 使用确定性算法并关闭 benchmark 模式

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
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="初始化一个检查点")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点保存目录路径")
    parser.add_argument("-pdm", "--pitch-dim-model", type=int, default=DEFAULT_PITCH_DIM_MODEL, help="音高特征编码器的主模型维度，默认为 %(default)s")
    parser.add_argument("-pdf", "--pitch-dim-feedforward", type=int, default=DEFAULT_PITCH_DIM_FEEDFORWARD, help="音高特征编码器前馈网络的隐藏层维度，默认为 %(default)s")
    parser.add_argument("-nh", "--num-heads", type=int, default=DEFAULT_NUM_HEADS, help="编-解码器注意力头的数量，默认为 %(default)s")
    parser.add_argument("-dh", "--dim-head", type=int, default=DEFAULT_DIM_HEAD, help="编-解码器每个注意力头的维度，默认为 %(default)s")
    parser.add_argument("-df", "--dim-feedforward", type=int, default=DEFAULT_DIM_FEEDFORWARD, help="编-解码器前馈网络的隐藏层维度，默认为 %(default)s")
    parser.add_argument("-pk1", "--pitch-kernel-size-1", type=int, default=DEFAULT_PITCH_CONV1_KERNEL, help="音高特征编码器前馈层第一个卷积核大小，默认为 %(default)s")
    parser.add_argument("-pk2", "--pitch-kernel-size-2", type=int, default=DEFAULT_PITCH_CONV2_KERNEL, help="音高特征编码器前馈层第二个卷积核大小，默认为 %(default)s")
    parser.add_argument("-vb", "--variance-bins", type=int, default=DEFAULT_VARIANCE_BINS, help="音符特征离散化的精细度，默认为 %(default)s")
    parser.add_argument("-pl", "--num-pitch-layers", type=int, default=DEFAULT_NUM_PITCH_LAYERS, help="音高特征编码器层数，默认为 %(default)s")
    parser.add_argument("-ncl", "--num-note-count-layers", type=int, default=DEFAULT_NUM_NOTE_COUNT_LAYERS, help="音符计数特征编码器层数，默认为 %(default)s")
    parser.add_argument("-pml", "--num-pitch-mean-layers", type=int, default=DEFAULT_NUM_PITCH_MEAN_LAYERS, help="音高均值特征编码器层数，默认为 %(default)s")
    parser.add_argument("-prl", "--num-pitch-range-layers", type=int, default=DEFAULT_NUM_PITCH_RANGE_LAYERS, help="音高范围特征编码器层数，默认为 %(default)s")
    parser.add_argument("-el", "--num-encoder-layers", type=int, default=DEFAULT_NUM_ENCODER_LAYERS, help="编码器层数，默认为 %(default)s")
    parser.add_argument("-dl", "--num-decoder-layers", type=int, default=DEFAULT_NUM_DECODER_LAYERS, help="解码器层数，默认为 %(default)s")
    parser.add_argument("-u", "--seed", default=8964, type=int, help="初始化检查点的种子，保证训练过程可复现，默认为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 检查注意力头的维度是否为偶数
    if args.dim_head % 2 == 1:
        raise RuntimeError("由于模型使用旋转位置编码，注意力头的维度必须为偶数。")

    # 设置随机种子，确保可复现性
    set_seed(args.seed)

    # 初始化模型
    model = MidiNet(MidiNetConfig(
        args.pitch_dim_model,
        args.pitch_dim_feedforward,
        args.num_heads,
        args.dim_head,
        args.dim_feedforward,
        args.pitch_kernel_size_1,
        args.pitch_kernel_size_2,
        args.variance_bins,
        args.num_pitch_layers,
        args.num_note_count_layers,
        args.num_pitch_mean_layers,
        args.num_pitch_range_layers,
        args.num_encoder_layers,
        args.num_decoder_layers,
    ))

    # 初始化优化器和梯度缩放器
    optimizer = optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", 1)

    # 删除之前的 SummaryWriter，为以后训练可视化模型初始化状态准备
    shutil.rmtree(args.ckpt_path / "logdir", ignore_errors=True)

    # 保存为检查点
    save_checkpoint(args.ckpt_path, model.state_dict(), optimizer.state_dict(), scaler.state_dict(), {
        "num_heads": args.num_heads,
        "completed_epochs": 0,
    })

    # 打印初始化成功信息
    print(f"检查点初始化成功，已保存到 {args.ckpt_path}")


if __name__ == "__main__":
    main(parse_args())
