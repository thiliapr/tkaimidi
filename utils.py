"训练和推理工具集: 包含MIDI处理、数据规范化和音高标准化等功能。"

# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import math
import mido
import time
import threading
from typing import TypeVar, Generic, Iterable, Iterator
from collections import Counter, deque

# 在非 Jupyter 环境下导入常量库
if "get_ipython" not in globals():
    from constants import NATURAL_SCALE, TIME_PRECISION, KEY_UP, KEY_DOWN, OCTAVE_JUMP_UP, OCTAVE_JUMP_DOWN, TIME_INTERVAL, LOOKAHEAD_COUNT

T = TypeVar("T")


class ThreadVariable(Generic[T]):
    """
    一个线程安全的变量容器，允许在多个线程之间安全地读取和修改值。

    该类使用锁来保护对值的访问，确保在任何给定时间只有一个线程可以修改该值。
    这避免了数据竞争和不一致的问题。

    Args:
        value: 初始值。
    """

    def __init__(self, value: T):
        self._value = value  # 初始化变量值
        self.lock = threading.Lock()  # 创建一个锁对象，用于线程同步

    @property
    def value(self) -> T:
        with self.lock:
            return self._value

    @value.setter
    def value(self, value: T):
        with self.lock:
            self._value = value


class BufferStream(Iterable, Generic[T]):
    """
    一个线程安全的迭代器，用于缓冲和逐步输出数据。
    支持多线程环境下的数据生产和消费，当缓冲区为空时会自动休眠等待。

    工作流程:
    1. 生产者线程通过send()方法添加数据到缓冲区
    2. 消费者通过迭代器逐个获取数据
    3. 当缓冲区为空时，迭代器会短暂休眠避免忙等待
    4. 调用stop()方法可以安全停止迭代器

    Returns:
        迭代器实例，支持直接迭代或调用其他方法

    Examples:
        >>> buffer_stream = BufferStream()
        >>> buffer_stream.send([1, 2, 3])
        >>> for item in buffer_stream:
        ...     print(item)
        ...     if item == 3: break
        >>> buffer_stream.stop()
    """

    def __init__(self):
        self.buffer = deque()  # 使用双端队列作为缓冲区
        self.lock = threading.Lock()  # 线程锁保证操作原子性
        self.stop_event = threading.Event()  # 停止标志位

    def send(self, data: list[T]):
        """
        向缓冲区添加数据，线程安全。

        Args:
            data: 要添加的整数列表

        Examples:
            >>> buffer_stream.send([1, 2, 3])
        """
        with self.lock:
            self.buffer.extend(data)

    def stop(self):
        """
        安全停止迭代器。
        设置停止标志位，迭代器将在下次检查时停止迭代。

        Examples:
            >>> buffer_stream.stop()
        """
        self.stop_event.set()

    def __iter__(self) -> Iterator[T]:
        """
        实现迭代器协议，返回生成器。
        当缓冲区不为空时返回数据，为空时短暂休眠。

        Returns:
            生成器对象，每次yield一个数据项

        Examples:
            >>> next(iter(buffer_stream))
        """
        while not self.stop_event.is_set():
            with self.lock:
                # 直接检查缓冲区是否非空
                if self.buffer:
                    # 成功获取数据后立即继续下一次迭代
                    yield self.buffer.popleft()
                    continue

            # 缓冲区为空时短暂休眠避免CPU占用过高
            time.sleep(0.1)


def midi_to_notes(midi_file: mido.MidiFile) -> list[tuple[int, int]]:
    """
    从给定的MIDI文件中提取音符信息并返回一个包含音符及其相对时间间隔的列表。

    本函数首先合并所有轨道，并按时间顺序整理所有MIDI消息。它会跳过打击乐通道（MIDI通道10以及其他被指定为打击乐的通道），
    只处理其他通道的力度不为0的`note_on`事件。接着，函数对音符的相对时间间隔除以时间精度并四舍五入，
    然后将相对时间除以它们的公因数来压缩时间，最后返回包含音符及其时间间隔的列表。

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
    compressed_intervals = [interval // gcd for interval in relative_intervals]

    # 去除重复的音符（相同音符与零时间间隔的重复）
    final_notes = []
    previous_note = None  # 上一个音符，用于避免重复

    for note, interval in zip(pitches, compressed_intervals):
        if interval == 0 and note == previous_note:
            continue  # 跳过重复的零间隔音符

        final_notes.append((note, interval))  # 添加音符和时间间隔
        previous_note = note  # 更新上一个音符

    return final_notes


def notes_to_sheet(notes: list[tuple[int, int]]) -> tuple[list[tuple[str, int]], list[int]]:
    """
    将MIDI音符列表转换为电子乐谱。

    Args:
        notes: MIDI音符列表，每个元组格式为(音高, 时间间隔)

    Returns:
        sheet: 电子乐谱事件列表，包含下列事件类型：
            - 0-11: 音符（音阶中的音高）。
            - 12: 音高下调一个半音。
            - 13: 音高上调一个半音。
            - 14: 音符下跳一个八度。
            - 15: 音符上跳一个八度。
            - 16: 时间间隔。
        positions: 每个音符在乐谱中的位置，记录每个音符出现的索引。
    """
    # 分离音高和时间间隔
    pitches, intervals = zip(*notes)

    # 定义最佳偏移量检测函数
    def offset_func(start: int, cur_offset: int) -> dict[int, int]:
        "计算分别在 0-11 的偏移量时有多少个音符在自然音阶。"
        end = min(start + LOOKAHEAD_COUNT, len(pitches))
        segment = [pitch + cur_offset for pitch in pitches[start:end]]
        return {offset: sum((pitch + offset) % 12 in NATURAL_SCALE for pitch in segment) for offset in range(12)}

    def octave_offset_func(start: int, cur_offset: int):
        "计算使音高集中在一个八度范围内的偏移量。"
        end = min(start + LOOKAHEAD_COUNT, len(pitches))
        segment = [pitch + cur_offset for pitch in pitches[start:end]]
        return -max(Counter(pitch // 12 for pitch in segment).items(), key=lambda x: (x[1], x[0] == 0))[0]

    # 计算调整音高的偏移量，使其尽量符合自然音阶
    cur_offset = max(offset_func(0, 0).items(), key=lambda x: x[1])[0]

    # 计算使音高集中在一个八度范围内的偏移量
    octave_offset = octave_offset_func(0, cur_offset)
    cur_offset += octave_offset * 12

    # 准备偏移量分数缓存
    offset_scores = offset_func(0, cur_offset)

    # 消除不参与分数计算的音符的影响
    if LOOKAHEAD_COUNT - 1 < len(pitches):
        for offset in range(12):
            if (pitches[LOOKAHEAD_COUNT - 1] + cur_offset + offset) % 12 in NATURAL_SCALE:
                offset_scores[offset] -= 1

    # 开始转换音符为电子乐谱
    sheet = []
    positions = []
    for i in range(len(pitches)):
        offset_sum = 0

        # 将最远能够看到的音符加入偏移量的分数计算
        if i + LOOKAHEAD_COUNT - 1 < len(pitches):
            for offset in range(12):
                if (pitches[i + LOOKAHEAD_COUNT - 1] + cur_offset + offset) % 12 in NATURAL_SCALE:
                    offset_scores[offset] += 1

        # 如果最佳偏移量不为 0，则调整偏移量并重新获取偏移量的分数
        best_offset = max(offset_scores.items(), key=lambda x: (x[1], x[0] == 0))[0]
        if best_offset != 0:
            offset_sum += best_offset
            cur_offset += best_offset
            offset_scores = offset_func(i, cur_offset)

        # 消除当前音符音高对分数的影响
        for offset in range(12):
            if (pitches[i] + cur_offset + offset) % 12 in NATURAL_SCALE:
                offset_scores[offset] -= 1

        # 如果音高不在一个八度范围内，调整音高
        best_octave_offset = octave_offset_func(i, cur_offset)
        if best_octave_offset != 0:
            offset_sum += best_octave_offset * 12
            cur_offset += best_octave_offset * 12

        # 如果有音高偏移，在乐谱中做标记
        if offset_sum:
            sheet.extend(KEY_UP if offset_sum > 0 else KEY_DOWN for _ in range(abs(offset_sum)))

        # 记录时间间隔
        if intervals[i]:
            sheet.extend(TIME_INTERVAL for _ in range(intervals[i]))

        # 记录乐谱中音符开始的位置
        positions.append(len(sheet))

        # 将当前音高调整到0-11范围内，并记录八度跳跃
        pitch = pitches[i] + cur_offset
        if pitch < 0 or pitch > 11:
            octave_jump = pitch // 12
            sheet.extend(OCTAVE_JUMP_UP if octave_jump > 0 else OCTAVE_JUMP_DOWN for _ in range(abs(octave_jump)))
            pitch %= 12

        # 记录音符
        sheet.append(pitch)

    # 返回结果
    return sheet, positions


def sheet_to_notes(sheet: Iterator[int]) -> Iterator[tuple[int, int, int]]:
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
        生成器，每次产生一个元组(音高, 时间间隔, 当前全局偏移)

    Examples:
        >>> list(sheet_to_notes(iter([0, 1, 14, 0])))
        [(0, 0, 0), (1, 0, 0), (-12, 0, 0)]
    """
    # 初始化状态变量
    global_offset = 0  # 当前全局偏移
    octave_offset = 0  # 当前八度偏移
    accumulated_interval = 0  # 累计时间间隔

    for event in sheet:
        if event < 12:
            # 计算并生成最终音符
            final_pitch = event - global_offset + octave_offset * 12
            yield final_pitch, accumulated_interval, global_offset

            # 重置状态
            octave_offset = accumulated_interval = 0
        elif event == 12:
            global_offset -= 1  # 降调
        elif event == 13:
            global_offset += 1  # 升调
        elif event == 14:
            octave_offset -= 1  # 降八度
        elif event == 15:
            octave_offset += 1  # 升八度
        else:
            accumulated_interval += 1  # 增加时间间隔


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
