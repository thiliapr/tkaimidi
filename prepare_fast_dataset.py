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
from utils.constants import PITCH_RANGE
from utils.midi import midi_to_notes


def convert(
    midi_files: list[pathlib.Path],
    max_frames: int,
    min_frames: int,
) -> list[np.ndarray]:
    """
    将 MIDI 文件集合转换为机器学习可用的数据集格式
    该函数读取 MIDI 文件，提取音符序列，并将结果转换为序列表示
    处理流程包括：读取 MIDI 文件、过滤不合格文件、提取音高、转换为序列格式

    Args:
        midi_files: MIDI文件路径列表
        max_frames: 允许的最大时间帧数，用于过滤过长的 MIDI 文件
        min_frames: 要求的最小时间帧数，用于过滤过短的 MIDI 文件

    Returns:
        包含所有 MIDI 文件数据列表，每个文件对应一个数组

    Examples:
        >>> dataset = convert(midi_files, 1989, 64)
        >>> len(dataset)
        8964
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

        # 提取音符序列
        notes = midi_to_notes(midi_file)

        # 过滤不符合要求的文件
        if not (min_frames <= len(notes) <= max_frames):
            continue

        # 提取音高
        pitches, _ = zip(*notes)

        # 将特征元组添加到数据集
        dataset.append(np.array(pitches) + PITCH_RANGE)

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
    parser.add_argument("--min-franes", type=int, default=64, help="要求的最小时间帧数，默认值为 %(default)s")
    parser.add_argument("--max-frames", type=int, default=4096, help="允许的最大时间帧数，默认值为 %(default)s")
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
    dataset = convert(midi_files, args.max_frames, args.min_frames)

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
        for task_id, sequence in enumerate(subset):
            data |= {f"{task_id}": sequence}
            length.append(len(sequence))

        # 将子集写入对应文件
        np.savez_compressed(args.output_dir / f"{filename}.npz", **data, length=np.array(length))
        print(f"数据集的 {proportion}/{total_proportion}，即 {len(subset)} 条数据，已保存到 {filename}.npz")


if __name__ == "__main__":
    main(parse_args())
