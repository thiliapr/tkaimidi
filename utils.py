# 训练和推理工具集
"""
训练和推理工具集: 包含MIDI处理、数据规范化和音高标准化等功能。
"""

# Copyright (C)  thiliapr 2024-2025
# License: AGPLv3-or-later

import math
import mido


def midi_to_notes(midi_file: mido.MidiFile) -> list[tuple[int, int, int, int]]:
    """
    将MIDI文件解析为结构化音符数据，支持多轨道处理和打击乐通道检测。

    Args:
        midi_file: 输入的MIDI文件对象

    Returns:
        音符列表，每个元组包含:
            - 通道号 (0-15)
            - 音高 (0-127)
            - 起始时间 (绝对时间，单位ticks)
            - 持续时间 (单位ticks)

    实现说明:
        1. 合并所有轨道并按时间顺序处理事件
        2. 自动检测打击乐通道 (从1开始数, 始终包含通道10)
        3. 支持MIDI Program Change事件动态更新打击乐通道
        4. 处理音符重叠和持续事件
    """
    notes = []
    drum_channels = {9}  # MIDI通道10(索引9)默认作为打击乐通道
    active_notes = {}  # 跟踪未结束的音符: {(channel, note): start_time}

    # 合并所有轨道并按时间排序
    merged_track = mido.merge_tracks(midi_file.tracks)
    now = 0  # 当前绝对时间(基于四分音符 480 ticks 的时基)

    for msg in merged_track:
        # 更新绝对时间(将delta时间转换为标准时基)
        now += msg.time * 480 // midi_file.ticks_per_beat

        # 处理音色变化事件 (动态更新打击乐通道)
        if msg.type == "program_change":
            if msg.channel == 9:
                continue  # 不对通道10做任何操作

            # 音色96-103和>=112为打击乐类
            if (96 <= msg.program <= 103) or msg.program >= 112:
                drum_channels.add(msg.channel)
            else:
                drum_channels.discard(msg.channel)

        # 跳过非音符事件和打击乐通道
        if not msg.type.startswith("note_") or msg.channel in drum_channels:
            continue

        key = (msg.channel, msg.note)

        # 处理音符结束(note_off 或 velocity=0 的 note_on)
        if (msg.type == "note_off" or msg.velocity == 0) and key in active_notes:
            start = active_notes.pop(key)
            notes.append((msg.channel, msg.note, start, now - start))

        # 处理音符开始
        elif msg.type == "note_on":
            active_notes[key] = now

    # 按起始时间排序并返回
    notes.sort(key=lambda x: x[2])
    return notes


def normalize_times(
    data: list[tuple[int, int]],
    time_precision: int,
    strict: bool = True
) -> list[tuple[int, int]]:
    """
    规范化时序数据，优化时间分布并适配模型输入要求。

    Args:
        data: 原始数据列表，元素为 (音高, 绝对时间)
        time_precision: 时间量化精度 (单位ticks)
        strict: 是否严格保持节奏特征

    Returns:
        规范化后的 (音高, 相对时间) 列表

    处理流程:
        1. 转换为相对时间序列
        2. 寻找最优时间缩放因子
        3. 时间量化与间隔修正
        4. 插入间隔音符并去重
    """
    # 分离音高和时间序列
    pitches, abs_times = zip(*data)

    # 转换为相对时间(时间间隔序列)
    rel_times = [abs_times[0]] + [abs_times[i] - abs_times[i - 1] for i in range(1, len(abs_times))]

    def calculate_time_loss(time_seq: list[float]) -> float:
        """
        计算时间序列的损失值，用于评估缩放因子质量。

        损失组成:
        - 量化误差: 时间值偏离量化网格的程度
        - 时间分布方差: 时间值的离散程度
        - 零时间惩罚: 过多相邻音符零间隔
        """
        quant_error = 0.0  # 量化误差
        variance = 0.0  # 时间方差
        zero_penalty = 0  # 零间隔计数
        max_gap_penalty = 0.0  # 大间隔惩罚

        mean = sum(time_seq) / len(time_seq)

        for t in time_seq:
            quant_error += 1 / (1 + math.exp(-abs(t / time_precision - round(t / time_precision))))  # 量化误差 (sigmoid加权)
            variance += (t - mean) ** 2  # 统计方差
            zero_penalty += int(t < time_precision / 2)  # 零间隔计数

        return quant_error * 0.5 + math.sqrt(variance / len(time_seq)) * 1.2 + max_gap_penalty + zero_penalty

    # 寻找最佳时间缩放因子
    best_scale, min_loss = 1, math.inf
    i = 9 if strict else 0  # 严格模式下，时间缩放因子从 1 开始算起 ( math: (9 + 1) / 10 = 1 )
    failed_counter = 0
    while True:
        scale = (i := i + 1) / 10
        tmp_times = [time * scale for time in rel_times]
        cur_loss = calculate_time_loss(tmp_times)
        if cur_loss < min_loss:
            best_scale = scale
            min_loss = cur_loss
            failed_counter = 0  # 重置失败计数
        elif failed_counter < 10:
            failed_counter += 1  # 增加失败计数
        else:
            break  # 超过失败次数，退出循环

    # 应用最佳缩放并量化
    processed_times = [round(t * best_scale / time_precision) * time_precision for t in rel_times]

    # 计算最大公约数来压缩时间轴
    time_gcd = math.gcd(*processed_times) if len(processed_times) > 0 else 1
    compressed_times = [t // time_gcd for t in processed_times]

    # 插入间隔音符并清理重复
    result = []
    prev_pitch = None
    for pitch, time in zip(pitches, compressed_times):
        # 跳过重复零间隔音符
        if time == 0 and pitch == prev_pitch:
            continue

        result.append((pitch, time))
        prev_pitch = pitch

    return result


def notes_to_note_intervals(notes: list[tuple[int, int]], interval: int) -> list[int]:
    """
    将MIDI音符列表转化为音符间隔格式。

    MIDI音符格式: [(音高, 与上一个音符的时间差), ...]
    音符间隔格式: [音高, 1个时间单位的停顿 (用`最大音高 + 1`表示), 音高, 音高, ...]

    举个例子 (假设最大音高是23):
    输入: `[(1, 0), (2, 3), (4, 0), (9, 0), (12, 2)]`
    输出: `[1, 24, 24, 24, 2, 4, 9, 24, 24, 12]`

    Args:
        notes: 原MIDI音符列表，包含音高和与上一个音符的时间差
        interval: 表示停顿的值

    Returns:
        音符间隔格式的列表
    """
    note_intervals = []  # 初始化音符间隔列表

    for pitch, time in notes:
        note_intervals.extend([interval] * time)  # 添加与上一个音符的时间差对应的停顿
        note_intervals.append(pitch)  # 添加当前音高

    return note_intervals  # 返回最终的音符间隔列表


def empty_cache():
    "清空 CUDA 显存缓存并执行垃圾回收。"
    import torch
    import gc

    # 检查是否有可用的 CUDA 设备
    if torch.cuda.is_available():
        # 仅在 CUDA 设备上调用 empty_cache()
        torch.cuda.empty_cache()

    # 执行 Python 垃圾回收
    gc.collect()
