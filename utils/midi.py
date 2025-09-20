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
    将音符和时间信息转换为 MIDI 轨道。
    该函数接受一个包含音符和相对时间间隔的列表，并生成一个 MIDI 轨道，其中包含音符开启和关闭事件。
    音符间隔格式为 (音高, 时间间隔)，时间间隔单位为 TIME_PRECISION。
    所有音符的持续时间都是 TIME_PRECISION。

    Args:
        notes: 音符间隔格式的列表

    Returns:
        包含音符事件及结束标记的 MIDI 轨道
    """
    # 生成 MIDI 事件队列
    events = []
    cumulative_time = 0  # 累计时间

    for pitch, interval in notes:
        # 计算事件绝对时间
        cumulative_time += interval * TIME_PRECISION

        # 添加音符开启和关闭事件
        events.append(("note_on", pitch, cumulative_time))
        events.append(("note_off", pitch, cumulative_time + TIME_PRECISION))

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
    从给定的MIDI文件中提取音符信息并返回一个包含音符及其相对时间间隔的列表。

    本函数首先合并所有轨道，并按时间顺序整理所有MIDI消息。它会跳过打击乐通道（MIDI通道10以及其他被指定为打击乐的通道），
    只处理其他通道的力度不为0的`note_on`事件。绝对时间的计算方式为：每条消息的`time`字段（单位为ticks）
    会根据MIDI文件的ticks_per_beat属性归一化到480 ticks每拍（即`current_time += msg.time * 480 // midi_file.ticks_per_beat`），
    这样可以保证不同MIDI文件的时间单位一致。接着，函数对音符的相对时间间隔除以时间精度并四舍五入，
    然后将相对时间除以它们的公因数来压缩时间，最后返回包含音符及其时间间隔的列表。
    根据经验，音符的持续时间对听感影响不大，因此在此函数中不考虑音符的持续时间。

    原理解释:
        假设有一个MIDI文件包含以下音符事件:
        - note 60 at time 0
        - note 67 at time 0  # 这是和弦
        - note 64 at time 480
        这表示音符60和67在同一时间开始，音符64在480 ticks后开始。

        计算时间间隔:
        - note 60: 0 - 0 = 0
        - note 67: 0 - 0 = 0
        - note 64: 480 - 0 = 480

        经过四舍五入和压缩后，返回的列表将是: [(60, 0), (67, 0), (64, 1)]

    Args:
        midi_file: 要提取的MIDI文件对象

    Returns:
        一个包含音符和相对时间间隔的列表，每个元素为一个元组(音高，时间间隔)。

    Examples:
        >>> midi_file = mido.MidiFile("example.mid")
        >>> midi_to_notes(midi_file)
        [(0, 0), (5, 1), (8, 1)]
    """
    extracted_notes = []  # 用于存储提取的音符信息（音高和相对时间）
    drum_channels = {9}  # 打击乐通道的集合，默认情况下，MIDI通道10（索引9）为打击乐通道
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
            # 将音符和相对时间添加到提取列表中
            extracted_notes.append((msg.note, current_time))

    # 如果没有提取出任何音符，则返回空列表
    if not extracted_notes:
        return []

    # 转化为相对时间
    extracted_notes = [(note, now - (extracted_notes[i - 1][1] if i else now)) for i, (note, now) in enumerate(extracted_notes)]

    # 提取音符和相对时间序列
    pitches, relative_intervals = zip(*extracted_notes)

    # 对时间间隔四舍五入
    relative_intervals = [int(interval / TIME_PRECISION + 0.5) for interval in relative_intervals]

    # 计算时间间隔的最大公约数，用于压缩时间轴
    gcd = math.gcd(*relative_intervals) if relative_intervals else 1

    # 如果所有音符都没有时间间隔，就返回当作有错误的 MIDI 并返回空列表
    if gcd == 0:
        return []

    # 压缩时间轴
    compressed_intervals = [interval // gcd for interval in relative_intervals]

    # 去除重复的音符（相同音符与零时间间隔的重复）
    deduped_notes = []
    previous_note = None  # 上一个音符，用于避免重复

    for note, interval in zip(pitches, compressed_intervals):
        if interval == 0 and note == previous_note:
            continue  # 跳过重复的零间隔音符

        deduped_notes.append((note, interval))  # 添加音符和时间间隔
        previous_note = note  # 更新上一个音符

    return deduped_notes


def notes_to_piano_roll(notes: list[tuple[int]]) -> np.ndarray:
    """
    将音符序列转换为钢琴卷帘矩阵表示。

    该函数接收一个包含音符和相对时间间隔的列表，将其转换为一个二维布尔矩阵（钢琴卷帘）。
    矩阵的行表示时间步，列表示音高（MIDI音符编号0-127），矩阵中的True值表示在对应时间点有音符开始。
    首先提取所有音符的音高和时间间隔，计算每个音符的绝对开始时间，然后创建一个全零矩阵，
    最后在对应的时间步和音高位置标记为 True。

    Args:
        notes: 包含音符和相对时间间隔的列表，每个元素为元组 (音高, 时间间隔)

    Returns:
        一个二维布尔数组，表示钢琴卷帘，形状为 [总时间步数, 128]

    Examples:
        >>> notes = [(60, 0), (67, 0), (64, 1)]
        >>> piano_roll = notes_to_piano_roll(notes)
        >>> piano_roll.shape
        (2, 128)
    """
    # 空音符列表时直接返回空钢琴卷帘
    if not notes:
        return np.empty((0, 128), dtype=bool)

    # 将音符列表解包为音高序列和时间间隔序列
    pitches, intervals = zip(*notes)

    # 计算累积时间得到每个音符的绝对时间
    start_times = np.cumsum(np.array(intervals))

    # 计算钢琴卷帘的总时间长度（最后一个音符的开始时间+1）
    total_time_steps = start_times[-1] + 1 if len(start_times) > 0 else 0

    # 初始化钢琴卷帘矩阵（时间步×音高）
    piano_roll = np.zeros((total_time_steps, 128), dtype=bool)

    # 在对应时间步和音高位置标记音符弹奏
    for pitch, time_step in zip(pitches, start_times):
        piano_roll[time_step, pitch] = True

    return piano_roll


def piano_roll_to_notes(piano_roll: np.ndarray) -> list[tuple[int, int]]:
    """
    将钢琴卷帘矩阵转换为音符序列。

    该函数接收一个二维布尔矩阵（钢琴卷帘），提取其中所有标记为 True 的位置，
    这些位置表示在对应时间步有音符开始。然后按时间步排序这些音符，计算每个音符
    与前一个音符之间的相对时间间隔，最后返回包含音高和相对时间间隔的列表。

    Args:
        piano_roll: 二维布尔数组，表示钢琴卷帘，形状为 [总时间步数, 128]

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
    previous_time = 0  # 记录上一个音符的时间步
    result = []  # 存储最终的音符序列

    # 遍历每个时间步
    for time, pitch_row in enumerate(piano_roll):
        for pitch in np.where(pitch_row)[0]:
            # 计算当前音符与上一个音符的相对时间间隔
            result.append((pitch, time - previous_time))

            # 更新上一个音符的时间步为当前时间步
            previous_time = time

    return result
