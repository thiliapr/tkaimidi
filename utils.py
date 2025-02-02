# 一些训练和推理会用到的工具
# Copyright (C)  thiliapr 2024-2025
# License: AGPLv3-or-later

import math
import mido


def midi_to_notes(midi_file: mido.MidiFile) -> list[tuple[int, int, int, int]]:
    """
    将 MIDI 文件转换为音符列表。

    Args:
        midi_file: 要解析的 MIDI 文件对象。

    Returns:
        list: 包含音符信息的列表，每个元组包含：
            - 通道号 (int)
            - 音高 (int)
            - 开始时间 (int) - 从MIDI文件开始到音符开始的时间
            - 持续时间 (int) - 音符持续的时间
    """
    notes = []  # 存储解析得到的音符信息
    drum_channels = {9}
    for track in midi_file.tracks:  # 遍历每个音轨
        now = 0  # 当前时间，从0开始
        pool: dict[tuple[int, int], int] = {}  # 存储音符的开始时间

        for message in track:  # 遍历音轨中的每个消息
            now += message.time * 480 // midi_file.ticks_per_beat  # 更新当前时间

            # 将通道标为打击乐器
            if message.type == "program_change" and message.channel != 9:
                if (96 <= message.program <= 103) or (112 <= message.program):
                    drum_channels.add(message.channel)
                else:
                    drum_channels.discard(message.channel)

            # 仅处理音符相关的消息
            if not message.type.startswith("note_"):
                continue
            # 去除打击乐器
            if message.channel in drum_channels:
                continue

            message_key = (message.channel, message.note)  # 唯一标识音符的键

            # 处理音符结束的情况
            if (message.type == "note_off" or message.velocity == 0) and message_key in pool:
                start_at = pool[message_key]  # 获取音符的开始时间
                # 添加音符信息到结果列表
                notes.append((message.channel, message.note, start_at, now - start_at))
                del pool[message_key]  # 从池中移除已结束的音符

            # 处理音符开始的情况
            elif message.type == "note_on":
                pool[message_key] = now  # 记录音符开始的时间

    # 根据音符的开始时间进行排序
    notes.sort(key=lambda info: info[2])
    return notes  # 返回解析得到的音符列表


def norm_data(data: list[tuple[int, int]], time_precision: int, max_time_diff: int, strict: bool = True) -> list[tuple[int, int]]:
    """
    规范化音符数据。

    Args:
        data: 输入的音符数据
            - 音符编号：音符的音高。
            - 开始时间：音符的开始时间。
        time_precision: 用于对齐音符开始时间的时间精度。所有开始时间都会四舍五入到该精度。
        max_time_diff: 两个音符之间允许的最大时间差。如果相邻音符之间的时间差大于此值，则会插入音符来填充时间。
        strict: 尽量保留MIDI原来的节奏

    Returns:
        list: 规范化后的音符数据，格式与输入相同，包含调整后的音符音高和时间。
    """
    def time_loss(times: list[int]) -> float:
        "计算时间损失，用于评估时间调整的效果。"
        precision_loss = 0  # 时间精度损失
        max_time_loss = 0  # 最大时间差损失
        variance = 0  # 时间标准差
        zero_time = 0  # 被四舍五入后无时间差音符的数量
        mean = sum(times) / len(times)  # 时间的平均值

        for time in times:
            precision_loss += 1 / (1 + math.exp(-abs(time / time_precision - math.ceil(time / time_precision)))) - 0.5  # 计算时间精度损失
            variance += (time - mean) ** 2  # 计算标准差
            zero_time += time < (time_precision // 2)

            # 计算最大时间差损失
            if time > max_time_diff:
                max_time_loss += 1 / (1 + math.exp(-math.floor(time / max_time_diff)))
        variance = (variance / len(times)) ** (1 / 2)  # 计算标准差

        loss_all = precision_loss + max_time_loss + variance * 1.2 + zero_time * 0.5
        return loss_all

    notes, times = (list(d) for d in zip(*data))  # 分离音符和时间
    lowest_dnote = min(notes)  # 计算音符的最低音高

    # 将音符降低到最低音域、计算每一个音符与前一个音符的开始时间差
    now = times[0]
    for i, (note, time) in enumerate(data):
        notes[i] = note - lowest_dnote
        times[i] -= now
        now = time

    # 寻找最佳的乘数以最小化时间损失
    best_multiple, best_multiple_loss = 1, math.inf
    i = 9 if strict else 0  # 严格模式下，时间乘数从 1 开始算起
    failed_counter = 0
    while True:
        cur_multiple = (i := i + 1) / 10
        tmp_times = [time * cur_multiple for time in times]
        cur_loss = time_loss(tmp_times)
        if cur_loss < best_multiple_loss:
            best_multiple = cur_multiple
            best_multiple_loss = cur_loss
            failed_counter = 0  # 重置失败计数
        elif failed_counter < 10:
            failed_counter += 1  # 增加失败计数
        else:
            break  # 超过失败次数，退出循环

    # 根据最佳乘数调整时间
    times = [time * best_multiple for time in times]

    # 按照时间精度四舍五入音符的开始时间
    for i, time in enumerate(times):
        div, mod = divmod(time, time_precision)
        times[i] = int(time_precision * (div + int(mod >= time_precision // 2)))

    # 将时间除以公因数
    time_gcd = math.gcd(*times) // time_precision
    times = [time // time_gcd for time in times]

    # 插入音符以填充时间差，并移除重复音符
    i = 1  # 索引指针
    while i < len(times):
        if times[i] > max_time_diff:
            # 插入音符以填充时间差
            notes.insert(i, notes[i])
            times.insert(i, max_time_diff)
            times[i + 1] -= max_time_diff
            i += 1
        elif notes[i - 1] == notes[i] and times[i] == 0:
            # 移除重复音符
            notes.pop(i)
            times.pop(i)
        else:
            i += 1

    times = [time // time_precision for time in times]  # 归一化
    return list(zip(notes, times))


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
