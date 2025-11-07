# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
import mido
import numpy as np
from utils.constants import TIME_PRECISION, PITCH_RANGE


def notes_to_track(notes: list[tuple[int, int]]) -> mido.MidiTrack:
    """
    将音符事件列表转换为 MIDI 轨道

    工作流程:
        1. 转化为绝对音高: 计算每个音符的绝对音高，并进行居中处理
        2. 生成 MIDI 事件队列: 为每个音符创建 note_on 和 note_off 事件，并按时间排序
        3. 构建 MIDI 轨道: 计算每个事件的相对时间，创建 MIDI 消息，并添加到轨道中
        4. 添加轨道结束标记

    Args:
        notes: 包含音符事件的列表，每个元素为 (相对音高, 相对时间) 的元组

    Returns:
        包含音符事件及结束标记的 MIDI 轨道
    """
    # 转化为绝对音高
    pitches, intervals = zip(*notes)
    pitches = np.array(pitches)
    pitches = np.cumsum(pitches)  # 计算绝对音高

    # 音高居中处理
    pitches += (127 - (pitches.max() + pitches.min())) // 2

    # 生成 MIDI 事件队列
    events = []
    current_time = 0  # 当前绝对时间
    for pitch, inteval in zip(pitches, intervals):
        # 计算当前绝对时间
        current_time += inteval

        # 添加音符开启和关闭事件
        events.append(("note_on", pitch, current_time * TIME_PRECISION))
        events.append(("note_off", pitch, (current_time + 1) * TIME_PRECISION))

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


def midi_to_notes(midi_file: mido.MidiFile) -> list[tuple[int, int]]:
    """
    从 MIDI 文件中提取音符信息，包括相对音高和相对时间
    处理过程包括：合并所有轨道、过滤打击乐音符、时间标准化、音高范围调整
    通过解析 MIDI 消息流，识别音符开始事件，排除打击乐通道，并对时间和音高进行标准化处理

    Args:
        midi_file: 输入的 MIDI 文件对象

    Returns:
        包含音符信息的列表，每个元素为 (相对音高, 相对时间) 的元组

    Examples:
        >>> midi = mido.MidiFile("example.mid")
        >>> notes = midi_to_notes(midi)
        >>> print(notes[:5])
        [(0, 0), (8, 1), (-9, 1), (6, 3), (-4, 1)]
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

    # 将音符整体前移，使第一个音符时间为 0
    times = [time - times[0] for time in times]

    # 对时间四舍五入
    times = [int(interval / TIME_PRECISION + 0.5) for interval in times]

    # 计算时间的最大公约数
    time_gcd = math.gcd(*times)

    # 如果时间全为 0，返回空列表
    if time_gcd == 0:
        return []

    # 压缩时间轴
    times = [time // time_gcd for time in times]

    # 每个时间只保留一个音符，方便提取主旋律
    notes = {}
    for pitch, time in zip(pitches, times):
        notes[time] = max(pitch, notes.get(time, -1))  # 保留音高最高的音符
    notes = sorted(notes.items())

    # 计算每个音符相对于之前音符的要音高和时间
    times, pitches = zip(*notes)
    pitches = [0] + [now - prev for prev, now in zip(pitches[:-1], pitches[1:])]
    intervals = [0] + [now - prev for prev, now in zip(times[:-1], times[1:])]

    # 如果音高变化幅度过大，则进行八度平移，限制在指定范围内
    adjusted_pitches = []
    octave_shift_accumulator = 0  # 累计八度偏移量
    for pitch in pitches:
        if abs(pitch) > PITCH_RANGE:
            # 如果音高变化超出范围，进行八度平移
            shift_direction = (1 if pitch < 0 else -1)  # 确定平移方向，音高为负则向上平移，音高为正则向下平移
            octaves_to_shift = math.ceil((abs(pitch) - PITCH_RANGE) / 12)  # 计算需要平移的八度数
            adjusted_pitches.append(pitch + shift_direction * octaves_to_shift * 12)  # 每个八度包含 12 个半音
            octave_shift_accumulator += shift_direction * octaves_to_shift  # 记录相比原音符偏离了多少个八度
        else:
            # 尝试恢复部分八度偏移
            shift_direction = 1 if octave_shift_accumulator < 0 else -1  # 确定平移方向，使累计偏移绝对值减小
            max_recoverable = min(int((PITCH_RANGE - shift_direction * pitch) / 12), abs(octave_shift_accumulator))
            adjusted_pitches.append(pitch + shift_direction * max_recoverable * 12)
            octave_shift_accumulator += shift_direction * max_recoverable

    # 构建最终音符列表
    notes = [(pitch, interval) for pitch, interval in zip(adjusted_pitches, intervals)]
    return notes


def notes_to_sequence(notes: list[tuple[int, int]]) -> list[int]:
    """
    将音符列表转换为序列表示
    
    将音符的 (相对音高, 时间间隔) 元组列表转换为整数序列，其中:
    - 0 表示时间间隔(休止符)
    - 非 0 值表示音符，值为 (音高 + pitch_range + 1)
    - 无论是音符还是休止符，他们都占据一个时间单位
    这种表示方法便于后续的序列处理和模型训练

    Args:
        notes: 音符列表，每个元素为 (相对音高, 时间间隔)

    Returns:
        整数序列，0 表示时间间隔，非 0 值表示音符开始

    Examples:
        >>> notes = [(0, 0), (5, 2), (-3, 1)]
        >>> notes_to_sequence(notes, 127)
        [128, 0, 133, 125]
    """
    sequence = []
    for pitch, interval in notes:
        # 添加时间间隔(用 0 表示)
        sequence.extend([0] * (interval - 1))
        # 添加音符开始标记，音高值偏移以避免与 0 冲突
        sequence.append(pitch + PITCH_RANGE + 1)
    return sequence


def sequence_to_notes(sequence: list[int]) -> list[tuple[int, int]]:
    """
    将序列表示转换回音符列表

    将整数序列反向转换为音符的 (相对音高, 时间间隔) 元组列表，
    这是 notes_to_sequence 的逆操作

    Args:
        sequence: 整数序列，0 表示休止符，非 0 值表示音符

    Returns:
        音符列表，每个元素为 (相对音高, 时间间隔)

    Examples:
        >>> sequence = [128, 0, 0, 133, 0, 125]
        >>> sequence_to_notes(sequence, 127)
        [(0, 0), (5, 2), (-3, 1)]
    """
    notes = []
    current_interval = 0  # 当前累计的时间间隔

    for value in sequence:
        if value == 0:
            # 遇到 0 表示时间间隔，累计间隔计数
            current_interval += 1
        else:
            # 遇到非 0 值表示音符开始，计算原始音高并添加音符
            original_pitch = value - PITCH_RANGE - 1
            notes.append((original_pitch, current_interval))
            current_interval = 1  # 重置时间间隔计数

    return notes
