"从 MIDI 文件夹中提取训练信息，以方便训练分词器和模型。"

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
    min_notes: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    将 MIDI 文件集合转换为机器学习可用的数据集格式
    处理流程包括：读取 MIDI 文件、过滤无效文件、转换为钢琴卷帘表示、提取音符统计特征，并最终打包为包含多种特征的数据集
    
    处理流程：
    1. 遍历所有 MIDI 文件路径
    2. 使用 mido 库读取并解析每个 MIDI 文件，跳过损坏文件
    3. 提取音符序列并过滤音符数量不足的文件
    4. 将音符序列转换为钢琴卷帘矩阵表示
    5. 过滤超过最大帧数限制的序列
    6. 计算每个时间帧的音符数量、平均音高和音高范围
    7. 将所有特征打包并添加到数据集中

    Args:
        midi_files: MIDI文件路径列表
        max_frames: 允许的最大时间帧数，超过此长度的序列将被过滤
        min_notes: 要求的最小音符数量，少于此刻数的文件将被跳过

    Returns:
        包含四个 NumPy 数组的元组列表：(钢琴卷帘矩阵, 音符数量数组, 平均音高数组, 音高范围数组)

    Examples:
        >>> dataset = convert(midi_files, 1000, 10)
        >>> len(dataset)
        50
    """
    dataset = []

    # 使用进度条显示处理进度
    for filepath in tqdm(midi_files):
        try:
            # 读取 MIDI 文件，clip=True 自动处理异常事件
            midi_file = mido.MidiFile(filepath, clip=True)
        except (ValueError, EOFError, OSError, mido.KeySignatureError):
            # 跳过无法解析的损坏文件
            continue

        # 提取音符序列并过滤音符数量不足的文件
        notes = midi_to_notes(midi_file)
        if len(notes) < min_notes:
            continue

        # 转换为钢琴卷帘表示 [时间帧, 128 个音高]
        piano_roll = notes_to_piano_roll(notes)
        if len(piano_roll) > max_frames:
            continue

        # 计算每个时间帧的音符数量并归一化
        note_counts = (piano_roll.sum(1) / 128).astype(np.float32)

        # 预分配数组存储音高统计特征
        pitch_means = np.empty(len(piano_roll), dtype=np.float32)
        pitch_ranges = np.empty(len(piano_roll), dtype=np.float32)

        # 遍历每个时间帧计算音高特征
        for time, pitch_row in enumerate(piano_roll):
            # 获取当前时间帧被触发的音高值
            active_pitches = np.where(pitch_row)[0]

            # 计算平均音高和音高范围并归一化
            if len(active_pitches) > 0:
                pitch_means[time] = active_pitches.mean() / 127
                pitch_ranges[time] = (active_pitches.max() - active_pitches.min()) / 127
            else:
                # 无音符时设置为特殊值 -1
                pitch_means[time] = -1
                pitch_ranges[time] = -1

        # 将特征元组添加到数据集
        dataset.append((piano_roll, note_counts, pitch_means, pitch_ranges))

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
    parser.add_argument("--max-frames", type=int, default=8964, help="钢琴卷帘表示中允许的最大时间帧数，默认值为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
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
    dataset = convert(midi_files, args.max_frames, args.min_notes)

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
