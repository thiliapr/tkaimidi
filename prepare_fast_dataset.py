# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
import random
from typing import Optional
import mido
import numpy as np
from tqdm import tqdm
from utils.midi import midi_to_notes, notes_to_piano_roll


def convert(
    midi_files: list[pathlib.Path],
    max_frames: int,
    min_notes: int,
    frame_length: int,
    smoothing_kernel_size: int,
    smoothing_sigma: float,
    smoothing_iterations: int,
    time_stretch_count: int,
    max_stretch_factor: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    将 MIDI 文件集合转换为机器学习可用的数据集格式

    处理流程包括读取 MIDI 文件、过滤无效文件、转换为钢琴卷帘表示、提取音符统计特征，
    并最终打包为包含多种特征的数据集。具体工作流程包括：
    1. 遍历所有 MIDI 文件路径，使用进度条显示处理进度
    2. 使用 mido 库读取并解析每个 MIDI 文件，自动跳过损坏或无法解析的文件
    3. 提取音符序列并过滤音符数量不足或时间过长的文件
    4. 对音符时间轴进行随机拉伸变换以增加数据多样性
    5. 将音符序列转换为钢琴卷帘矩阵表示
    6. 预分配数组存储音符统计特征
    7. 使用滑动窗口计算每个时间帧的音符数量、平均音高和音高范围
    8. 使用卷积核对特征进行平滑处理
    9. 基于整个数据集的统计信息对特征进行归一化处理

    Args:
        midi_files: MIDI文件路径列表
        max_frames: 允许的最大时间帧数，用于过滤过长的 MIDI 文件
        min_notes: 要求的最小音符数量，用于过滤音符过少的 MIDI 文件
        frame_length: 滑动窗口的帧长度，用于计算局部特征
        smoothing_kernel_size: 高斯平滑卷积核的大小
        smoothing_sigma: 高斯平滑的标准差，控制平滑程度
        smoothing_iterations: 特征平滑的迭代次数
        time_stretch_count: 时间拉伸变换的采样数量
        max_stretch_factor: 最大时间拉伸因子

    Returns:
        包含四个 NumPy 数组的元组列表，分别表示钢琴卷帘矩阵、音符数量数组、平均音高数组和音高范围数组

    Examples:
        >>> dataset = convert(midi_files, 1989, 64, 16, 2)
        >>> len(dataset)
        8964
    """
    dataset = []

    # 准备高斯卷积核
    smoothing_kernel = np.exp(-0.5 * (np.arange(smoothing_kernel_size) - (smoothing_kernel_size - 1) / 2) ** 2 / smoothing_sigma ** 2)

    # 使用进度条显示处理进度
    for filepath in tqdm(midi_files):
        try:
            # 读取 MIDI 文件，clip=True 自动处理异常事件
            midi_file = mido.MidiFile(filepath, clip=True)
        except (ValueError, EOFError, OSError, mido.KeySignatureError):
            # 跳过无法解析的损坏文件
            continue

        # 提取音符序列并过滤不符合要求的文件
        notes = midi_to_notes(midi_file, 87)
        if len(notes) < min_notes or notes[-1][1] > max_frames:
            continue

        # 提取音高和时间信息
        pitches, times = zip(*notes)

        # 生成时间拉伸因子范围，用于数据增强
        stretch_factors = range(1, min(max_frames // times[-1], max_stretch_factor) + 1)

        # 随机选择指定数量的拉伸因子进行数据增强
        for stretch_factor in random.sample(stretch_factors, k=min(time_stretch_count, len(stretch_factors))):
            # 应用时间拉伸变换
            stretched_times = [time * stretch_factor for time in times]
            stretched_notes = list(zip(pitches, stretched_times))

            # 转换为钢琴卷帘表示 [时间帧, 88 个音高]
            piano_roll = notes_to_piano_roll(stretched_notes)

            # 预分配特征数组
            note_counts = np.empty(len(piano_roll), dtype=np.float32)
            pitch_means = np.empty(len(piano_roll), dtype=np.float32)
            pitch_ranges = np.empty(len(piano_roll), dtype=np.float32)

            # 填充钢琴卷帘矩阵以便处理边界情况，使用中值填充减少异常值影响
            padded_roll = np.pad(piano_roll, (((frame_length - 1) // 2, (frame_length - 1) // 2), (0, 0)), "median")

            # 预计算每个时间帧的激活音高索引，提高后续处理效率
            active_pitches = [np.where(padded_roll[time])[0] for time in range(len(padded_roll))]

            # 初始化上一次的有效值（用于处理空帧情况）
            last_mean = last_range = 0

            # 遍历每个时间帧计算音高特征
            for time in range(len(padded_roll) - frame_length + 1):
                # 合并窗口内的所有音高
                window_pitches = np.concatenate(active_pitches[time:time + frame_length])

                # 计算音符数量（未归一化）
                note_counts[time] = len(window_pitches)

                # 计算平均音高和音高范围（使用标准差作为范围度量）
                if len(window_pitches):
                    pitch_means[time] = last_mean = window_pitches.mean()
                    pitch_ranges[time] = last_range = window_pitches.std()
                else:
                    # 如果没有激活音高，使用上一次的有效值保持连续性
                    pitch_means[time] = last_mean
                    pitch_ranges[time] = last_range

            # 使用卷积平滑特征曲线，方便神经网络训练
            for _ in range(smoothing_iterations):
                note_counts, pitch_means, pitch_ranges = [
                    np.convolve(np.pad(feature, ((smoothing_kernel_size - 1) // 2, (smoothing_kernel_size - 1) // 2), "edge"), smoothing_kernel, "valid")
                    for feature in [note_counts, pitch_means, pitch_ranges]
                ]

            # 将特征元组添加到数据集
            dataset.append([piano_roll, note_counts, pitch_means, pitch_ranges])

    # 计算最小最大值用于归一化
    note_count_min, note_count_max = np.percentile(np.concatenate([data[1] for data in dataset]), [1, 99])
    pitch_mean_min, pitch_mean_max = np.percentile(np.concatenate([data[2] for data in dataset]), [1, 99])
    pitch_range_min, pitch_range_max = np.percentile(np.concatenate([data[3] for data in dataset]), [1, 99])

    # 归一化音符数量、平均音高、音高范围
    for idx, (_, note_counts, pitch_means, pitch_ranges) in enumerate(dataset):
        dataset[idx][1] = ((note_counts - note_count_min) / (note_count_max - note_count_min + 1e-8)).astype(np.float32)
        dataset[idx][2] = ((pitch_means - pitch_mean_min) / (pitch_mean_max - pitch_mean_min + 1e-8)).astype(np.float32)
        dataset[idx][3] = ((pitch_ranges - pitch_range_min) / (pitch_range_max - pitch_range_min + 1e-8)).astype(np.float32)

    return dataset


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="准备专用于特定检查点训练的数据集，用于加快训练时数据加载速度")
    parser.add_argument("dataset", type=pathlib.Path, help="数据集的路径。该文件夹应包含 MIDI 格式数据")
    parser.add_argument("output_dir", type=pathlib.Path, help="处理后的特征数据集输出目录")
    parser.add_argument("splits", type=str, nargs="+", help="输出文件名和拆分比例，格式为`filename:proportion`，如`train:9`和`val:1`")
    parser.add_argument("--min-notes", type=int, default=64, help="MIDI 文件中至少包含的音符数量，默认值为 %(default)s")
    parser.add_argument("--max-frames", type=int, default=4096, help="钢琴卷帘表示中允许的最大时间帧数，默认值为 %(default)s")
    parser.add_argument("--frame-length", type=int, default=33, help="用于计算音符统计特征的滑动窗口帧长度，默认值为 %(default)s")
    parser.add_argument("--smoothing-kernel-size", type=int, default=33, help="高斯平滑卷积核的大小，必须为奇数，默认值为 %(default)s")
    parser.add_argument("--smoothing-sigma", type=float, default=1.0, help="高斯平滑的标准差，控制平滑程度，默认值为 %(default)s")
    parser.add_argument("--smoothing-iterations", type=int, default=16, help="用于平滑音符统计特征的卷积迭代次数，默认值为 %(default)s")
    parser.add_argument("--time-stretch-count", type=int, default=4, help="每个 MIDI 文件随机采样的时间拉伸变换数量，默认值为 %(default)s")
    parser.add_argument("--max-stretch-factor", type=int, default=4, help="时间拉伸的最大因子，默认值为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 验证提取配置
    if args.frame_length < 0 or args.frame_length % 2 == 0:
        raise Exception("滑动窗口帧长度 --frame-length 必须正数且奇数")
    if args.smoothing_kernel_size < 0 or args.smoothing_kernel_size % 2 == 0:
        raise Exception("高斯平滑卷积核的大小 --smoothing-kernel-size 必须正数且奇数")

    # 遍历数据集文件
    midi_files = [
        file
        for file in args.dataset.rglob("*")
        if file.is_file() and file.suffix.lower() in {".mid", ".midi"}
    ]

    # 解析拆分配置
    splits = [str_split.split(":", 1) for str_split in args.splits]
    splits = [(filename, int(proportion)) for filename, proportion in splits]
    total_proportion = sum(proportion for _, proportion in splits)

    # 验证数据集大小是否满足拆分需求
    if len(midi_files) < total_proportion:
        raise RuntimeError(f"MIDI 数量（{len(midi_files)}）小于指定的拆分比例总和（{total_proportion}）")

    # 打印开始转换消息
    print(f"正在转换 {len(midi_files)} 个 MIDI 文件 ...")

    # 预处理数据
    dataset = convert(midi_files, args.max_frames, args.min_notes, args.frame_length, args.smoothing_kernel_size, args.smoothing_sigma, args.smoothing_iterations, args.time_stretch_count, args.max_stretch_factor)

    # 验证数据集大小是否满足拆分需求
    # 这里再次验证是因为之前预处理可能过滤了一部分无效数据，导致总数据量减少
    if len(dataset) < total_proportion:
        raise RuntimeError(f"有效 MIDI 数量（{len(dataset)}）小于指定的拆分比例总和（{total_proportion}）")

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 随机打乱数据
    random.shuffle(dataset)

    # 按比例拆分数据
    split_data = [dataset[rank::total_proportion] for rank in range(total_proportion)]

    # 写入拆分后的文件
    for filename, proportion in splits:
        # 从拆分数据中提取对应比例的数据
        subset = [item for chunk in split_data[:proportion] for item in chunk]
        split_data = split_data[proportion:]

        # 转换为字典形式，并记录序列的长度
        data = {}
        length = []
        for task_id, (piano_roll, note_counts, pitch_means, pitch_ranges) in enumerate(subset):
            data |= {
                f"{task_id}:piano_roll": piano_roll,
                f"{task_id}:note_counts": note_counts,
                f"{task_id}:pitch_means": pitch_means,
                f"{task_id}:pitch_ranges": pitch_ranges,
            }
            length.append(len(piano_roll))

        # 将子集写入对应文件
        np.savez_compressed(args.output_dir / f"{filename}.npz", **data, length=np.array(length))
        print(f"数据集的 {proportion}/{total_proportion}，即 {len(subset)} 条数据，已保存到 {filename}.npz")


if __name__ == "__main__":
    main(parse_args())
