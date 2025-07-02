"训练和推理工具集: 包含MIDI处理、数据规范化和音高标准化等功能。"

# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

from collections.abc import Callable
import math
from typing import Iterator, Optional
import mido
from constants import NATURAL_SCALE, TIME_PRECISION, KEY_UP, KEY_DOWN, OCTAVE_JUMP_UP, OCTAVE_JUMP_DOWN, TIME_INTERVAL, LOOKAHEAD_COUNT


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


def notes_to_sheet(notes: list[tuple[int, int]], max_length: Optional[int] = None) -> tuple[list[tuple[str, int]], list[int]]:
    """
    将MIDI音符列表转换为电子乐谱（事件序列）。

    Args:
        notes: MIDI音符列表，每个音符是一个元组 (pitch, interval)：
            - pitch: 音高（MIDI音符编号，0-127）
            - interval: 时间间隔（单位：ticks或自定义时间单位）
        max_length: 乐谱的最大允许长度（事件数量）。如果超过，则截断。

    Returns:
        返回包括两个值的数组:
            - sheet: 电子乐谱事件列表，包含以下事件类型：
                - 0-11: 音符（音阶中的音高，0=C, 1=C#, ..., 11=B）
                - 12 (KEY_DOWN): 音高下调一个半音
                - 13 (KEY_UP): 音高上调一个半音
                - 14 (OCTAVE_DOWN): 音符下移一个八度
                - 15 (OCTAVE_UP): 音符上移一个八度
                - 16 (TIME_INTERVAL): 时间间隔
            - positions: 每个音符在 sheet 中的索引位置列表
    """
    # 如果指定了最大长度，截断音符序列
    if max_length:
        notes = notes[:max_length]

    # 分离音高和时间间隔
    pitches, intervals = zip(*notes)

    # 定义最佳偏移量检测函数
    def calculate_natural_scale_matches(start_idx: int, current_pitch_offset: int) -> dict[int, int]:
        "计算不同音高偏移量下，音符符合自然音阶的数量。"
        end = min(start_idx + LOOKAHEAD_COUNT, len(pitches))
        segment = [pitch + current_pitch_offset for pitch in pitches[start_idx:end]]
        return {offset: sum((pitch + offset) % 12 in NATURAL_SCALE for pitch in segment) for offset in range(12)}

    def calculate_octave_offset(start_idx: int, current_pitch_offset: int):
        "计算最佳八度偏移，使音符集中在合理的八度范围内。"
        end = min(start_idx + LOOKAHEAD_COUNT, len(pitches))
        segment = [pitch + current_pitch_offset for pitch in pitches[start_idx:end]]
        segment_mean = sum(segment) / len(segment)

        # 如果平均音高偏离中央区域(±18半音)太远，则调整八度
        return -int(segment_mean / 12) if abs(segment_mean) > 18 else 0

    # 初始音高偏移: 选择使前 LOOKAHEAD_COUNT 个音符最匹配自然音阶的偏移量
    current_pitch_offset = max(calculate_natural_scale_matches(0, 0).items(), key=lambda x: x[1])[0]

    # 调整八度使音符集中在合理范围内
    octave_offset = calculate_octave_offset(0, current_pitch_offset)
    current_pitch_offset += octave_offset * 12

    # 初始化自然音阶匹配分数缓存
    natural_scale_scores = calculate_natural_scale_matches(0, current_pitch_offset)

    # 预先移除即将离开滑动窗口的音符对分数的影响
    if LOOKAHEAD_COUNT - 1 < len(pitches):
        for offset in range(12):
            if (pitches[LOOKAHEAD_COUNT - 1] + current_pitch_offset + offset) % 12 in NATURAL_SCALE:
                natural_scale_scores[offset] -= 1

    # 开始转换每个音符
    sheet = []  # 乐谱事件序列
    positions = []  # 每个音符在`sheet`中的位置
    for i in range(len(pitches)):
        total_pitch_adjustment = 0  # 当前位置需要调整的总半音数

        # 添加新进入窗口的音符到分数计算
        if i + LOOKAHEAD_COUNT - 1 < len(pitches):
            for offset in range(12):
                if (pitches[i + LOOKAHEAD_COUNT - 1] + current_pitch_offset + offset) % 12 in NATURAL_SCALE:
                    natural_scale_scores[offset] += 1

        # 选择最佳音高调整（分数相同时，优先选择0偏移）
        best_pitch_adjustment = max(natural_scale_scores.items(), key=lambda x: (x[1], x[0] == 0))[0]
        if best_pitch_adjustment != 0:
            total_pitch_adjustment += best_pitch_adjustment
            current_pitch_offset += best_pitch_adjustment
            natural_scale_scores = calculate_natural_scale_matches(i, current_pitch_offset)

        # 移除当前处理音符对分数的影响（因为它即将离开窗口）
        for offset in range(12):
            if (pitches[i] + current_pitch_offset + offset) % 12 in NATURAL_SCALE:
                natural_scale_scores[offset] -= 1

        # 检查是否需要八度调整
        optimal_octave_shift = calculate_octave_offset(i, current_pitch_offset)
        if optimal_octave_shift != 0:
            total_pitch_adjustment += optimal_octave_shift * 12
            current_pitch_offset += optimal_octave_shift * 12

        # 检查乐谱长度限制
        if max_length:
            # 音高调整事件 + 时间间隔事件 + 八度跳跃事件 + 音符本身
            events_needed = abs(total_pitch_adjustment) + intervals[i] + abs((pitches[i] + current_pitch_offset) // 12) + 1
            if len(sheet) + events_needed > max_length:
                break

        # 添加音高调整事件到乐谱
        if total_pitch_adjustment:
            sheet.extend(KEY_UP if total_pitch_adjustment > 0 else KEY_DOWN for _ in range(abs(total_pitch_adjustment)))

        # 记录时间间隔
        if intervals[i]:
            sheet.extend(TIME_INTERVAL for _ in range(intervals[i]))

        # 记录乐谱中音符开始的位置
        # 由于八度跳跃是修正音符的八度位置的，音符包括八度跳跃，所以需要在音符的八度跳跃前记录位置
        positions.append(len(sheet))

        # 将当前音高调整到[0, 11]范围内，并记录八度跳跃
        pitch = pitches[i] + current_pitch_offset
        if pitch < 0 or pitch > 11:
            octave_jump = pitch // 12
            sheet.extend(OCTAVE_JUMP_UP if octave_jump > 0 else OCTAVE_JUMP_DOWN for _ in range(abs(octave_jump)))
            # 归一化 pitch 到 [0, 11] 范围
            pitch %= 12

        # 记录音符
        sheet.append(pitch)

    # 返回结果
    return sheet, positions


def sheet_to_notes(sheet: Iterator[int]) -> Iterator[tuple[int, int]]:
    """
    将电子乐谱转换为MIDI音符序列。

    该函数处理输入的电子乐谱数据流，将其转换为MIDI音符序列。转换规则如下：
    1. 数值0-11表示音符，会结合当前偏移量计算最终音高
    2. 数值12-15用于控制音高偏移量
    3. 其他数值累加为时间间隔
    转换过程会维护全局偏移和八度偏移状态，并自动重置时间间隔

    Args:
        sheet: 电子乐谱数据流，由notes_to_sheet()生成

    Returns:
        生成器，每次产生一个元组(音高, 时间间隔)

    Examples:
        >>> list(sheet_to_notes([0, 1, 14, 0]))
        [(0, 0), (1, 0), (-12, 0)]
    """
    # 初始化状态变量
    global_offset = 0  # 当前全局偏移
    octave_offset = 0  # 当前八度偏移
    accumulated_interval = 0  # 累计时间间隔

    for event in sheet:
        if event < 12:
            # 计算并生成最终音符
            final_pitch = event - global_offset + octave_offset * 12
            yield final_pitch, accumulated_interval

            # 重置状态
            octave_offset = accumulated_interval = 0
        elif event == KEY_DOWN:
            global_offset -= 1  # 降调
        elif event == KEY_UP:
            global_offset += 1  # 升调
        elif event == OCTAVE_JUMP_DOWN:
            octave_offset -= 1  # 降八度
        elif event == OCTAVE_JUMP_UP:
            octave_offset += 1  # 升八度
        else:
            accumulated_interval += 1  # 增加时间间隔


def parallel_map(func: Callable, iterable: list[tuple], num_workers: Optional[int] = None):
    """
    使用多进程并行执行函数。
    该函数将给定的函数应用于可迭代对象的每个元素，并使用指定数量的工作进程并行处理。

    Args:
        func: 要应用的函数，接受可迭代对象的元素作为参数
        iterable: 可迭代对象，包含要处理的数据
        num_workers: 工作进程数量，默认为CPU核心数

    Returns:
        包含函数应用结果的列表
    """
    from multiprocessing import Pool, cpu_count
    num_workers = num_workers or cpu_count()  # 获取CPU核心数或指定数量
    with Pool(processes=num_workers) as pool:
        return pool.starmap(func, iterable)


def empty_cache():
    """
    清理 CUDA 显存缓存并执行 Python 垃圾回收。

    本函数会先触发 Python 的垃圾回收机制，释放未被引用的内存。
    如果检测到有可用的 CUDA 设备，则进一步清理 CUDA 显存缓存，释放未被 PyTorch 占用但已缓存的 GPU 显存。

    Examples:
        >>> empty_cache()
    """
    import torch
    import gc

    # 执行 Python 垃圾回收
    gc.collect()

    # 检查是否有可用的 CUDA 设备
    if torch.cuda.is_available():
        # 仅在 CUDA 设备上调用 empty_cache()
        torch.cuda.empty_cache()
