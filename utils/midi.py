# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
import mido
import numpy as np
from utils.constants import TIME_PRECISION


def notes_to_track(notes: list[tuple[int, int]]) -> mido.MidiTrack:
    """
    将音符事件列表转换为 MIDI 轨道

    工作流程:
        1. 生成 MIDI 事件队列: 为每个音符创建 note_on 和 note_off 事件，并按时间排序
        2. 构建 MIDI 轨道: 计算每个事件的相对时间，创建 MIDI 消息，并添加到轨道中
        3. 添加轨道结束标记

    Args:
        notes: 包含音符事件的列表，每个元素为 (音高, 时间) 的元组

    Returns:
        包含音符事件及结束标记的 MIDI 轨道
    """
    # 生成 MIDI 事件队列
    events = []
    for pitch, time in notes:
        # 添加音符开启和关闭事件
        events.append(("note_on", pitch, time * TIME_PRECISION))
        events.append(("note_off", pitch, (time + 1) * TIME_PRECISION))

    # 按事件发生时间排序，若时间相同则 note_on 优先于 note_off
    events.sort(key=lambda x: (x[2], x[0] == "note_off"))

    # 构建 MIDI 轨道
    track = [mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(128))]
    last_event_time = 0  # 上一个事件的绝对时间

    for event_type, note, event_time in events:
        # 计算相对于上一个事件的时间差
        delta_ticks = event_time - last_event_time

        # 创建 MIDI 消息
        track.append(mido.Message(
            event_type,
            note=note,
            velocity=100 if event_type == "note_on" else 0,  # 音符力度
            time=delta_ticks,  # 相对时间
            channel=0  # MIDI 通道
        ))

        # 更新最后事件时间
        last_event_time = event_time

    # 添加轨道结束标记
    return mido.MidiTrack(track + [mido.MetaMessage("end_of_track")])


def midi_to_notes(midi_file: mido.MidiFile, pitch_range: int = 127) -> list[tuple[int, int]]:
    """
    从 MIDI 文件中提取音符信息，包括音高和时间
    处理过程包括：合并所有轨道、过滤打击乐音符、时间标准化、音高范围调整
    通过解析 MIDI 消息流，识别音符开始事件，排除打击乐通道，并对时间和音高进行标准化处理

    Args:
        midi_file: 输入的 MIDI 文件对象
        pitch_range: 音高范围

    Returns:
        包含音符信息的列表，每个元素为 (音高, 时间) 的元组

    Examples:
        >>> midi = mido.MidiFile("example.mid")
        >>> notes = midi_to_notes(midi)
        >>> print(notes[:5])
        [(24, 0), (44, 1), (64, 2), (84, 3), (104, 4)]
    """
    notes = []  # 用于存储提取的音符信息（音高和时间）
    drum_channels = {9}  # 打击乐通道的集合，默认情况下，MIDI 通道 10（索引 9）为打击乐通道
    current_time = 0  # 当前的绝对时间

    # 合并所有轨道并按时间顺序排序
    merged_track = mido.merge_tracks(midi_file.tracks)
    for msg in merged_track:
        # 计算绝对时间
        current_time += msg.time * 480 // midi_file.ticks_per_beat

        # 处理音色变化事件，动态更新打击乐通道
        if msg.type == "program_change":
            if msg.channel == 9:
                continue  # 打击乐通道的音色变化消息忽略

            # 判断音色是否属于打击乐类别，音色范围: 96-103 或 >= 112
            if (96 <= msg.program <= 103) or msg.program >= 112:
                drum_channels.add(msg.channel)
            else:
                drum_channels.discard(msg.channel)

        # 跳过打击乐通道，处理其他通道的音符开始事件
        elif msg.type == "note_on" and msg.velocity != 0 and msg.channel not in drum_channels:
            # 将音符和时间添加到提取列表中
            notes.append((msg.note, current_time))

    # 如果没有提取出任何音符，则返回空列表
    if not notes:
        return []

    # 提取音符和时间序列
    pitches, times = zip(*notes)

    # 使第一个音符一开始就播放
    times = [time - times[0] for time in times]

    # 对时间四舍五入
    times = [int(interval / TIME_PRECISION + 0.5) for interval in times]

    # 计算时间的最大公约数，压缩时间轴
    time_gcd = math.gcd(*times)
    times = [time // time_gcd for time in times]

    # 去除重复的音符
    notes = sorted({(pitch, time) for pitch, time in zip(pitches, times)}, key=lambda x: x[1])

    # 将音高平移至 0 基准
    pitches, times = zip(*notes)
    pitches = np.array(pitches)
    pitches -= pitches.min()

    # 如果音高超出范围，进行压缩调整
    if pitches.max() > pitch_range:
        # 压缩音高范围到 [-inf, pitch_range]
        # 使音高最大值为 pitch_range，由于之前 pitches.min() 为 0，音高最小值一定会变成负数
        pitches = pitches - pitches.max() + pitch_range

        # 将音高向上平移，直到音高最小值为 0，选择能使最多音符有效的偏移量
        pitches += max(
            range(1 - pitches.min()),
            key=lambda offset: ((0 <= (pitches + offset)) & ((pitches + offset) <= pitch_range)).sum()
        )

    # 重新构建音符列表
    notes = [
        (pitch.item(), time)
        for pitch, time in zip(pitches, times)
        if 0 <= pitch <= pitch_range
    ]

    # 音高居中处理
    pitches, times = zip(*notes)
    pitches = np.array(pitches)
    pitches -= pitches.min()
    pitches += (pitch_range - pitches.max()) // 2

    # 构建最终音符列表
    notes = [
        (pitch.item(), time)
        for pitch, time in zip(pitches, times)
    ]

    return notes


def notes_to_piano_roll(notes: list[tuple[int]]) -> np.ndarray:
    """
    将音符序列转换为钢琴卷帘矩阵表示。

    该函数接收一个包含音符和相对时间间隔的列表，将其转换为一个二维布尔矩阵（钢琴卷帘）。
    矩阵的行表示时间步，列表示音高（钢琴 88 个音），矩阵中的 True 值表示在对应时间点有音符开始。
    首先提取所有音符的音高和时间间隔，计算每个音符的绝对开始时间，然后创建一个全零矩阵，
    最后在对应的时间步和音高位置标记为 True。

    Args:
        notes: 包含音符和相对时间间隔的列表，每个元素为元组 (音高, 时间间隔)

    Returns:
        一个二维布尔数组，表示钢琴卷帘，形状为 [总时间步数, 88]

    Examples:
        >>> notes = [(60, 0), (67, 0), (64, 1)]
        >>> piano_roll = notes_to_piano_roll(notes)
        >>> piano_roll.shape
        (2, 88)
    """
    # 空音符列表时直接返回空钢琴卷帘
    if not notes:
        return np.empty((0, 88), dtype=bool)

    # 将音符列表解包为音高序列和时间间隔序列
    _, times = zip(*notes)

    # 初始化钢琴卷帘矩阵（时间步×音高）
    piano_roll = np.zeros((times[-1] + 1, 88), dtype=bool)

    # 在对应时间步和音高位置标记音符弹奏
    for pitch, time_step in notes:
        piano_roll[time_step, pitch] = True

    return piano_roll


def piano_roll_to_notes(piano_roll: np.ndarray, pitch_offset: int = 20) -> list[tuple[int, int]]:
    """
    将钢琴卷帘矩阵转换为音符序列。

    该函数接收一个二维布尔矩阵（钢琴卷帘），提取其中所有标记为 True 的位置，
    这些位置表示在对应时间步有音符开始。然后按时间步排序这些音符，计算每个音符
    与前一个音符之间的相对时间间隔，最后返回包含音高和相对时间间隔的列表。

    Args:
        piano_roll: 二维布尔数组，表示钢琴卷帘，形状为 [总时间步数, 128]
        pitch_offset: 音高偏移量，默认为 20，将音高值向上平移以适应钢琴 88 键范围

    Returns:
        一个包含音符和相对时间间隔的列表，每个元素为元组(音高，时间间隔)

    Examples:
        >>> piano_roll = np.zeros((2, 128), dtype=bool)
        >>> piano_roll[0, 60] = True
        >>> piano_roll[0, 67] = True
        >>> piano_roll[1, 64] = True
        >>> piano_roll_to_notes(piano_roll)
        [(60, 0), (67, 0), (64, 1)]
    """
    result = []  # 存储最终的音符序列
    # 遍历每个时间步
    for time, pitch_row in enumerate(piano_roll):
        for pitch in np.where(pitch_row)[0]:
            result.append((pitch + pitch_offset, time))
    return result
