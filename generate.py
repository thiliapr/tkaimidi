"生成MIDI文件"
# Copyright (C)  thiliapr 2024-2025
# License: AGPLv3-or-later

import math
import pathlib
import mido
import tqdm
import torch
import torch.nn.functional as F

# 内置音乐曲库
LOVE_TRADING_MIDI = [(45, 0), (76, 0), (52, 1), (57, 1), (81, 0), (59, 1), (60, 1), (84, 0), (64, 1), (84, 1), (76, 2), (81, 2), (57, 1), (69, 1), (84, 0), (64, 1), (60, 1), (43, 2), (83, 0), (50, 1), (55, 1), (83, 0), (57, 1), (59, 1), (81, 0), (62, 1), (79, 1), (76, 2), (55, 3), (67, 1), (62, 1), (59, 1), (55, 1), (41, 1), (74, 0), (48, 1), (53, 1), (74, 0), (55, 1), (57, 1), (74, 0), (60, 1), (69, 1), (74, 2), (76, 2), (53, 1), (65, 1), (79, 0), (60, 1), (57, 1), (43, 2), (76, 0), (50, 1), (55, 1), (83, 0), (57, 1), (59, 1), (83, 0), (62, 1), (79, 1), (76, 2), (74, 3), (71, 1), (67, 1), (62, 1), (64, 1), (45, 1), (76, 0), (52, 1), (57, 1), (81, 0), (59, 1), (60, 1), (84, 0), (64, 1), (84, 1), (76, 2), (81, 2), (57, 1), (69, 1), (84, 0), (64, 1), (60, 1), (43, 2), (88, 0), (50, 1), (55, 1), (88, 0), (57, 1), (59, 1), (86, 0), (62, 1), (84, 1), (86, 2), (55, 3), (67, 1), (62, 1), (59, 1), (55, 1), (41, 1), (88, 0), (48, 1), (53, 1), (55, 1), (88, 0), (57, 1), (86, 0), (60, 1), (84, 1), (43, 2), (86, 0), (50, 1), (55, 1), (57, 1), (86, 0), (59, 1), (84, 0), (62, 1), (83, 1), (45, 2), (79, 0), (52, 1), (57, 1), (76, 0), (59, 1), (60, 1), (79, 0), (64, 1), (81, 1), (81, 2), (76, 3), (72, 1), (69, 1), (64, 1), (69, 1)]

# 在非Jupyter环境下导入模型库
if "get_ipython" not in globals():
    from model import MidiNet, TIME_PRECISION, MAX_NOTE, load_checkpoint
    from utils import notes_to_note_intervals


def model_output_to_track(note_intervals: list[int]) -> mido.MidiTrack:
    """
    将音符和时间信息转换为 MIDI 轨道。

    Args:
        note_intervals: 音符间隔格式的列表

    Returns:
        包含音符事件及结束标记的 MIDI 轨道
    """
    # 生成 MIDI 事件队列
    events = []
    cumulative_time = 0  # 累计时间

    for event in note_intervals:
        if event == MAX_NOTE + 1:
            # 计算事件绝对时间
            cumulative_time += TIME_PRECISION
        else:
            # 添加音符开启和关闭事件
            events.append(("note_on", event, cumulative_time))
            events.append(("note_off", event, cumulative_time + TIME_PRECISION))

    # 按事件发生时间排序（确保事件顺序正确）
    events.sort(key=lambda x: x[2])

    # 构建 MIDI 轨道
    track = [mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(128))]
    last_event_time = 0  # 上一个事件的绝对时间

    for event_type, note, event_time in events:
        # 计算相对于上一个事件的时间差（delta time）
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


def normalize_pitches(pitches: list[int]) -> tuple[list[int], int]:
    """
    音高标准化处理，确保所有音高在 [0, MAX_NOTE] 范围内。

    Args:
        pitches: 原始音高列表

    Returns:
        标准化后的音高列表, 总偏移量

    处理步骤:
        1. 整体平移使最低音为0
        2. 按八度降低超出范围的音高
        3. 中心化音高分布
    """
    if not pitches:
        return [], 0

    # 最低音对齐到0
    min_pitch = min(pitches)
    base_shifted = [p - min_pitch for p in pitches]

    # 处理超出上限的音高
    for i, p in enumerate(base_shifted):
        if p > MAX_NOTE:
            base_shifted[i] -= math.ceil((p + 1 - MAX_NOTE) / 12) * 12  # 降低若干个八度数使音高处于范围之中

    # 使音高范围居中
    current_max = max(base_shifted)
    center_shift = (MAX_NOTE - current_max) // 2
    final_pitches = [p + center_shift for p in base_shifted]

    return final_pitches, (center_shift - min_pitch)


def generate_midi(
    prompt: list[tuple[int, int]],
    model: MidiNet,
    seed: int,
    length: int,
    temperature: float = 1,
    top_p: float = 0.8
) -> mido.MidiTrack:
    """
    使用模型生成 MIDI 音乐轨道

    Args:
        model: 用于生成的神经网络模型
        prompt: 初始音符序列，格式为 [(音高, 时间单位), ...]
        seed: 随机种子
        length: 需要生成的音符数量
        temperature: 采样温度（大于1增加多样性，小于1减少随机性）
        top_p: 核采样概率阈值，范围是0至1

    Returns:
        生成的 MIDI 轨道
    """
    original_pitches, original_times = zip(*prompt)  # 预处理提示序列
    normalized_pitches, pitch_offset = normalize_pitches(original_pitches)  # 标准化音高范围

    # 将音符和时间编码为模型输入序列
    input_prompt = notes_to_note_intervals(zip(normalized_pitches, original_times), MAX_NOTE + 1)

    model.eval()  # 将模型设置为评估模式
    generator = torch.Generator().manual_seed(seed)  # 固定随机种子，确保每次生成结果一致

    # 生成循环
    for i in tqdm.tqdm(range(length), desc="Generate MIDI"):
        # 准备模型输入
        input_tensor = torch.tensor(input_prompt, dtype=torch.long).unsqueeze(0)

        # 获取模型预测
        with torch.no_grad():
            logits = model(input_tensor, mask=False)[0, -1, :]

        # 应用温度缩放
        probs = F.softmax(logits / temperature, dim=-1)

        # 核采样
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs <= top_p
        mask[..., -1:] = True  # 确保至少选择一个

        # 从截断分布中采样
        filtered_probs = sorted_probs * mask
        filtered_probs /= filtered_probs.sum()
        sampled_index = torch.multinomial(filtered_probs, 1, generator=generator)

        # 映射回原始索引
        selected_token = sorted_indices[sampled_index]
        input_prompt.append(selected_token.item())

    # 解码生成的序列
    decoded_note_intervals = [
        event if (event == MAX_NOTE + 1) else (event - pitch_offset)  # 补偿音高偏移
        for event in input_prompt
    ]

    return model_output_to_track(decoded_note_intervals)  # 转化为 MIDI 轨道


def main():
    model = MidiNet()
    try:
        state = load_checkpoint(pathlib.Path("ckpt"), train=False)  # 加载模型的预训练检查点
        model.load_state_dict(state)
    except Exception as e:
        print(f"Error in LoadCKPT: {e}")
    mido.MidiFile(tracks=[generate_midi(
        prompt=LOVE_TRADING_MIDI,
        model=model,
        seed=42,
        length=256)
    ]).save("example.mid")


if __name__ == "__main__":
    main()
