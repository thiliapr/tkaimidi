"这个模块实现了 MidiNet 模型的生成和处理功能，包括音乐生成、音符转换等。"

# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
from typing import Optional
from matplotlib import pyplot as plt
import mido
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from utils.checkpoint import load_checkpoint
from utils.model import MidiNet
from utils.midi import midi_to_notes, notes_to_piano_roll, piano_roll_to_notes, notes_to_track


@torch.inference_mode
def generate(model: MidiNet, prompt: torch.Tensor, num_frames: int, show_progress: bool = True):
    """
    使用 MidiNet 模型逐步生成音乐帧序列。
    
    该函数通过自回归方式生成指定数量的音乐帧。每步生成一帧，并将该帧作为下一时间步的输入。
    同时记录生成过程中的音符数量、音高均值和音高范围等统计信息。
    生成过程使用 KV Cache 优化，避免重复计算。

    Args:
        model: 用于音乐生成的神经网络模型
        prompt: 初始提示序列，钢琴卷帘，形状为[批次大小, 序列长度, 128]
        num_frames: 需要生成的帧数
        show_progress: 是否显示进度条

    Returns:
        生成的完整序列、音符数量预测、音高均值预测、音高范围预测

    Examples:
        >>> prompt = torch.randn(1, 10, 128)
        >>> output, note_counts, pitch_means, pitch_ranges = generate(model, prompt, 100)
    """
    # 在提示序列前添加全零起始帧
    prompt = torch.cat([torch.zeros([prompt.size(0), 1, 128], dtype=torch.float32, device=prompt.device), prompt.to(dtype=torch.float32)], dim=1)

    # 初始化 KV Cache 和预测结果容器
    kv_cache = None
    note_count_preds = torch.empty(1, 0, device=prompt.device)
    pitch_mean_preds = torch.empty(1, 0, device=prompt.device)
    pitch_range_preds = torch.empty(1, 0, device=prompt.device)

    # 逐步生成 num_frames 帧
    for _ in tqdm(range(num_frames), disable=not show_progress):
        # 首次使用完整提示，后续仅使用最后一帧
        model_input = prompt if kv_cache is None else prompt[:, -1:]
        note_pred, note_count_pred, pitch_mean_pred, pitch_range_pred, kv_cache = model(
            F.sigmoid(model_input) > 0.5,
            kv_cache=kv_cache
        )

        # 收集预测结果
        note_count_preds = torch.cat([note_count_preds, note_count_pred], dim=1)
        pitch_mean_preds = torch.cat([pitch_mean_preds, pitch_mean_pred], dim=1)
        pitch_range_preds = torch.cat([pitch_range_preds, pitch_range_pred], dim=1)

        # 将预测添加到序列中
        prompt = torch.cat([prompt, note_pred], dim=1)
    
    return F.sigmoid(prompt[:, 1:]), note_count_preds, pitch_mean_preds, pitch_range_preds


def plot_piano_roll(piano_roll: np.ndarray, pitch_mean: np.ndarray, pitch_range: np.ndarray, ax: plt.Axes):
    """
    绘制钢琴卷帘可视化图表，展示音符分布和音高趋势

    该函数创建一个综合可视化界面，包含两个主要部分：
    1. 钢琴卷帘网格：用矩形块表示每个时间点上激活的音符
    2. 音高统计曲线：显示平均音高和音高范围的变化趋势

    工作流程：
    - 设置坐标轴范围和基本参数
    - 遍历钢琴卷帘矩阵，为每个激活的音符绘制矩形块
    - 绘制平均音高曲线和音高范围填充区域

    Args:
        piano_roll: 钢琴卷帘数据，形状为 [时间步数, 128]
        pitch_mean: 每个时间步的平均音高
        pitch_range: 每个时间步的音高范围
        ax: Matplotlib 坐标轴对象，用于绘制图形

    Returns:
        无返回值，直接在输入的坐标轴上绘制图形

    Examples:
        >>> fig, ax = plt.subplots(figsize=(12, 6))
        >>> plot_piano_roll(piano_roll_data, pitch_mean, pitch_range, ax)
        >>> plt.show()
    """
    # 统计最大、最小音高，并绘制钢琴卷帘音符
    max_pitch, min_pitch = 0, 127
    for time, pitch_row in enumerate(piano_roll):
        for pitch in np.where(pitch_row > 0.5)[0]:
            # 统计音高
            max_pitch = max(max_pitch, int(pitch))
            min_pitch = min(min_pitch, int(pitch))

            # 为每个激活的音符绘制矩形块
            ax.add_patch(plt.Rectangle((time, pitch - 0.4), 1, 0.8, facecolor="skyblue", edgecolor="black", alpha=0.9 * pitch_row[pitch]))

    # 设置坐标轴范围
    ax.set_xlim(0, len(piano_roll))
    ax.set_ylim(min_pitch - 0.5, max_pitch + 0.5)

    # 创建共享 x 轴的第二个 y 轴
    varaince_ax = ax.twinx()

    # 绘制平均音高曲线
    pitch_x = np.arange(len(piano_roll)) + 0.5  # 使音高均值对齐音符
    varaince_ax.plot(pitch_x, pitch_mean, label="Pitch Mean", color="blue", alpha=0.5)

    # 绘制音高范围填充区域
    varaince_ax.fill_between(
        pitch_x,
        pitch_mean - pitch_range / 2,
        pitch_mean + pitch_range / 2,
        alpha=0.2
    )

    # 创造图例
    varaince_ax.legend(loc="upper right")

    # 设置标签
    ax.set_xlabel("Time Step")
    ax.set_ylabel("MIDI Note Number")
    varaince_ax.set_ylabel("Pitch Statistics")


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 可选的命令行参数列表，默认为 None，表示使用 sys.argv

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description="以指定 MIDI 为前面部分并生成音乐和保存。")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点的路径")
    parser.add_argument("num_frames", type=int, help="需要生成的帧数")
    parser.add_argument("output_path", type=pathlib.Path, help="MIDI 文件保存路径。生成的 MIDI 文件将会保存到这里。")
    parser.add_argument("-m", "--midi-path", type=pathlib.Path, help="指定的 MIDI 文件，将作为生成的音乐的前面部分。如果未指定，将从头开始生成。")
    parser.add_argument("-s", "--show-piano-roll", action="store_true", help="是否显示生成过程中音高均值和音高范围的钢琴卷帘图表")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型的预训练检查点
    state_dict, model_config, _ = load_checkpoint(args.ckpt_path)

    # 初始化模型并加载状态
    model = MidiNet(model_config)
    model.load_state_dict(state_dict)

    # 转移模型到设备并设置为评估模式
    model = model.to(device).eval()

    # 加载音乐生成的提示部分
    prompt_notes = []  # 默认空提示音符列表
    if args.midi_path:
        try:
            prompt_notes = midi_to_notes(mido.MidiFile(args.midi_path))
        except Exception as e:
            print(f"加载指定的 MIDI 文件时出错: {e}\n将使用空提示音符生成")

    # 转换为钢琴卷轴、添加批次维度并开始生成
    piano_roll, _, pitch_means, pitch_ranges = generate(
        model,
        torch.tensor(notes_to_piano_roll(prompt_notes), device=device).unsqueeze(0),
        args.num_frames
    )

    # 删除批次维度并转化为 NumPy 数组
    piano_roll, pitch_means, pitch_ranges = [x.squeeze(0).cpu().numpy() for x in (piano_roll, pitch_means, pitch_ranges)]
    
    # 如果需要，绘制频率图表
    if args.show_piano_roll:
        _, ax = plt.subplots(figsize=(12, 6))
        plot_piano_roll(piano_roll, pitch_means, pitch_ranges, ax)
        plt.show()

    # 转换为 MIDI 轨道并保存为文件
    track = notes_to_track(piano_roll_to_notes(piano_roll))
    mido.MidiFile(tracks=[track]).save(args.output_path)


if __name__ == "__main__":
    main(parse_args())
