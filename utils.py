# 训练和推理工具集
"""
训练和推理工具集: 包含MIDI处理、数据规范化和音高标准化等功能。
"""

# Copyright (C)  thiliapr 2024-2025
# License: AGPLv3-or-later

import math
import mido

NATURAL_SCALE = {0, 2, 4, 5, 7, 9, 11}


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
    interval_precision: int,
    strict: bool = True
) -> list[tuple[int, int]]:
    """
    规范化时序数据，优化时间分布并适配模型输入要求。

    Args:
        data: 原始数据列表，元素为 (音高, 绝对时间)
        interval_precision: 时间间隔量化精度 (单位ticks)
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
    rel_intervals = [abs_times[0]] + [abs_times[i] - abs_times[i - 1] for i in range(1, len(abs_times))]

    def calculate_interval_loss(interval_seq: list[float]) -> float:
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

        mean = sum(interval_seq) / len(interval_seq)

        for t in interval_seq:
            quant_error += 1 / (1 + math.exp(-abs(t / interval_precision - round(t / interval_precision))))  # 量化误差 (sigmoid加权)
            variance += (t - mean) ** 2  # 统计方差
            zero_penalty += int(t < interval_precision / 2)  # 零间隔计数

        return quant_error * 0.5 + math.sqrt(variance / len(interval_seq)) * 1.2 + max_gap_penalty + zero_penalty

    # 寻找最佳时间缩放因子
    best_scale, min_loss = 1, math.inf
    i = 9 if strict else 0  # 严格模式下，时间缩放因子从 1 开始算起 ( math: (9 + 1) / 10 = 1 )
    failed_counter = 0
    while True:
        scale = (i := i + 1) / 10
        tmp_intervals = [interval * scale for interval in rel_intervals]
        cur_loss = calculate_interval_loss(tmp_intervals)
        if cur_loss < min_loss:
            best_scale = scale
            min_loss = cur_loss
            failed_counter = 0  # 重置失败计数
        elif failed_counter < 10:
            failed_counter += 1  # 增加失败计数
        else:
            break  # 超过失败次数，退出循环

    # 应用最佳缩放并量化
    processed_intervals = [round(t * best_scale / interval_precision) * interval_precision for t in rel_intervals]

    # 计算最大公约数来压缩时间轴
    interval_gcd = math.gcd(*processed_intervals) if len(processed_intervals) > 0 else 1
    compressed_intervals = [t // interval_gcd for t in processed_intervals]

    # 插入间隔音符并清理重复
    result = []
    prev_pitch = None
    for pitch, interval in zip(pitches, compressed_intervals):
        # 跳过重复零间隔音符
        if interval == 0 and pitch == prev_pitch:
            continue

        result.append((pitch, interval))
        prev_pitch = pitch

    return result


def notes_to_sheet(notes: list[tuple[int, int]], check_notes: int = 64) -> list[tuple[str, int]]:
    """
    将MIDI音符列表转换为电子乐谱，通过调整音高使其尽可能符合自然音阶并集中在一个八度范围内。

    Args:
        notes: MIDI音符列表，每个元组格式为(音高, 时间间隔)
        check_notes: 调整音高时检查后续音符的数量。

    Returns:
        电子乐谱事件列表，包含以下事件类型:
            - note: 音符 (0-11表示音阶中的音高)
            - key_shift: 全局音高偏移的半音数
            - octave_jump: 一个音符跳跃的八度数
            - interval: 两个音符的时间间隔
    """
    # 分离音高和时间间隔
    pitches, intervals = (list(item) for item in zip(*notes))

    # 将音高调整为相对最小音高的偏移
    min_pitch = min(pitches)
    pitches = [pitch - min_pitch for pitch in pitches]

    # 调整音高使其尽量落在自然音阶
    offset, _ = max(
        [(offset, sum(((pitch + offset) % 12) in NATURAL_SCALE for pitch in pitches[:check_notes])) for offset in range(12)],
        key=lambda x: x[1]
    )
    pitches = [pitch + offset for pitch in pitches]

    # 调整音高使其尽量集中在一个八度范围内
    octave_offset, _ = max(
        [(octave_offset, sum(0 <= (pitch + octave_offset * 12) < 12 for pitch in pitches[:check_notes])) for octave_offset in range(-2, 2)],
        key=lambda x: x[1]
    )
    pitches = [pitch + octave_offset * 12 for pitch in pitches]

    # 开始转化
    sheet: list[tuple[str, int]] = []
    for i in range(len(pitches)):
        original_pitch = pitches[i]
        offset_sum = 0

        # 如果当前音高不在自然音阶内，调整音高
        if (original_pitch % 12) not in NATURAL_SCALE:
            offset, _ = max(
                [(offset, sum(((pitch + offset) % 12) in NATURAL_SCALE for pitch in pitches[i:i + check_notes]) * check_notes + int(offset == 0)) for offset in range(12)],
                key=lambda x: x[1]
            )
            offset_sum += offset
            pitches = [pitch + offset for pitch in pitches]

        # 如果当前音高不在一个八度范围内，调整音高
        if original_pitch < 0 or original_pitch > 11:
            octave_offset, _ = max(
                [(octave_offset, sum(0 <= (pitch + octave_offset * 12) < 12 for pitch in pitches[i:i + check_notes]) * check_notes + int(octave_offset == 0)) for octave_offset in range(-2, 2)],
                key=lambda x: x[1]
            )
            offset_sum += octave_offset * 12
            pitches = [pitch + octave_offset * 12 for pitch in pitches]

        # 如果有音高偏移，在乐谱中做标记
        if offset_sum:
            sheet.append(("key_shift", offset_sum))

        # 将当前音高调整到0-11范围内，并记录八度跳跃
        cur_pitch = pitches[i]
        if cur_pitch < 0 or cur_pitch > 11:
            sheet.append(("octave_jump", cur_pitch // 12))
            cur_pitch %= 12

        # 记录音符和时间间隔
        if intervals[i]:
            sheet.append(("interval", intervals[i]))
        sheet.append(("note", cur_pitch))

    # 返回结果
    return sheet


def sheet_to_notes(sheet: list[tuple[str, int]]) -> list[tuple[int, int]]:
    """
    将电子乐谱转换为MIDI音符列表。

    Args:
        sheet: 由`notes_to_sheet(notes)`生成的电子乐谱

    Returns:
        MIDI音符列表，每个元组格式为(音高, 时间间隔)
    """
    notes = []
    global_offset = 0  # 全局音高偏移
    octave_offset = 0  # 八度偏移
    interval_sum = 0  # 累计时间间隔

    for event, value in sheet:
        if event == "key_shift":
            global_offset += value  # 更新全局音高偏移
        elif event == "octave_jump":
            octave_offset += value  # 更新八度偏移
        elif event == "note":
            # 计算最终音高并添加到音符列表
            final_pitch = value - global_offset + octave_offset * 12
            notes.append([final_pitch, interval_sum])
            octave_offset = interval_sum = 0  # 重置八度偏移和累计时间间隔
        elif event == "interval":
            interval_sum += value
        else:
            raise ValueError(f"Unknown event type: {event}")

    # 调整音高，使其最小音高为0
    min_pitch = min(pitch for pitch, _ in notes)
    notes = [(pitch - min_pitch, interval) for pitch, interval in notes]

    return notes


def sheet_to_model(sheet: list[tuple[str, int]]) -> list[int]:
    """
    将电子乐谱转换为模型的输入/输出格式。

    Args:
        sheet: 由`notes_to_sheet(notes)`生成的电子乐谱

    Returns:
        模型数据:
            - 0-11: 音符 (0-11表示音阶中的音高)
            - 12-23: 全局音高向下偏移的半音数。如果乐谱中小于-12，被转换成多个向下跳跃。
            - 24-35: 全局音高向上偏移的半音数。如果乐谱中大于12，被转换成多个向下跳跃。
            - 36-37: 一个音符向下跳跃的八度数。如果乐谱中小于-2，被转换成多个向下跳跃。
            - 38-39: 一个音符向上跳跃的八度数。如果乐谱中大于2，被转换成多个向上跳跃。
            - 40: 时间间隔单位。乐谱中的时间间隔(值大于1)，被转换成多个时间间隔单位。
    """
    model_data = []

    for event, value in sheet:
        if event == "key_shift":
            # 处理全局音高偏移
            if value < 0:
                # 向下偏移
                while value < -12:
                    model_data.append(23)
                    value += 12
                if value:
                    model_data.append(11 - value)  # 12-23表示全局音高向下偏移1-12个半音
            else:
                # 向上偏移
                while value > 12:
                    model_data.append(35)
                    value -= 12
                if value:
                    model_data.append(23 + value)  # 24-35表示全局音高向上偏移1-12个半音
        elif event == "octave_jump":
            # 处理八度跳跃
            if value < 0:
                # 向下跳跃
                while value < -2:
                    model_data.append(37)
                    value += 2
                if value:
                    model_data.append(35 - value)  # 36-37表示一个音符向下跳跃1-2个八度
            else:
                # 向上跳跃
                while value > 2:
                    model_data.append(39)
                    value -= 2
                if value:
                    model_data.append(37 + value)  # 38-39表示一个音符向上跳跃1-2个八度
        elif event == "note":
            # 处理音符
            model_data.append(value)  # 0-11表示音阶中的音高
        elif event == "interval":
            # 处理时间间隔
            for _ in range(value):
                model_data.append(40)  # 40表示时间间隔单位
        else:
            raise ValueError(f"Unknown event type: {event}")

    return model_data


def model_to_sheet(model_data: list[int]) -> list[tuple[str, int]]:
    """
    将模型的输入/输出格式转换回电子乐谱。

    Args:
        model_data: 由`sheet_to_model(sheet)`生成的模型数据

    Returns:
        电子乐谱
    """
    sheet = []
    i = 0
    n = len(model_data)

    while i < n:
        value = model_data[i]

        if 0 <= value <= 11:
            # 处理音符
            sheet.append(("note", value))
        elif 12 <= value <= 23:
            # 处理全局音高向下偏移
            offset = 11 - value
            sheet.append(("key_shift", offset))
        elif 24 <= value <= 35:
            # 处理全局音高向上偏移
            offset = value - 23
            sheet.append(("key_shift", offset))
        elif 36 <= value <= 37:
            # 处理一个音符向下跳跃的八度数
            octave_jump = 35 - value
            sheet.append(("octave_jump", octave_jump))
        elif 38 <= value <= 39:
            # 处理一个音符向上跳跃的八度数
            octave_jump = value - 37
            sheet.append(("octave_jump", octave_jump))
        elif value == 40:
            # 处理时间间隔单位
            interval_count = 1
            while i + 1 < n and model_data[i + 1] == 40:
                interval_count += 1
                i += 1
            sheet.append(("interval", interval_count))
        else:
            raise ValueError(f"Unknown model data value: {value}")

        i += 1

    return sheet


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
