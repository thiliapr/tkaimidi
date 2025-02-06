# 生成MIDI文件
# Copyright (C)  thiliapr 2024-2025
# License: AGPLv3-or-later

import pathlib
import mido
import tqdm
import torch
import torch.nn.functional as F

# 内置音乐曲库
LOVE_TRADING_MIDI = [(45, 0), (76, 0), (52, 1), (57, 1), (81, 0), (59, 1), (60, 1), (84, 0), (64, 1), (84, 1), (76, 2), (81, 2), (57, 1), (69, 1), (84, 0), (64, 1), (60, 1), (43, 2), (83, 0), (50, 1), (55, 1), (83, 0), (57, 1), (59, 1), (81, 0), (62, 1), (79, 1), (76, 2), (55, 3), (67, 1), (62, 1), (59, 1), (55, 1), (41, 1), (74, 0), (48, 1), (53, 1), (74, 0), (55, 1), (57, 1), (74, 0), (60, 1), (69, 1), (74, 2), (76, 2), (53, 1), (65, 1), (79, 0), (60, 1), (57, 1), (43, 2), (76, 0), (50, 1), (55, 1), (83, 0), (57, 1), (59, 1), (83, 0), (62, 1), (79, 1), (76, 2), (74, 3), (71, 1), (67, 1), (62, 1), (64, 1), (45, 1), (76, 0), (52, 1), (57, 1), (81, 0), (59, 1), (60, 1), (84, 0), (64, 1), (84, 1), (76, 2), (81, 2), (57, 1), (69, 1), (84, 0), (64, 1), (60, 1), (43, 2), (88, 0), (50, 1), (55, 1), (88, 0), (57, 1), (59, 1), (86, 0), (62, 1), (84, 1), (86, 2), (55, 3), (67, 1), (62, 1), (59, 1), (55, 1), (41, 1), (88, 0), (48, 1), (53, 1), (55, 1), (88, 0), (57, 1), (86, 0), (60, 1), (84, 1), (43, 2), (86, 0), (50, 1), (55, 1), (57, 1), (86, 0), (59, 1), (84, 0), (62, 1), (83, 1), (45, 2), (79, 0), (52, 1), (57, 1), (76, 0), (59, 1), (60, 1), (79, 0), (64, 1), (81, 1), (81, 2), (76, 3), (72, 1), (69, 1), (64, 1), (69, 1)]

# 在非Jupyter环境下导入模型库
if "get_ipython" not in globals():
    from model import MidiNet, TIME_PRECISION, NOTE_DURATION_COUNT, load_checkpoint


def model_output_to_track(notes: list[tuple[int, int]]) -> mido.MidiTrack:
    """
    将音符和时间信息转换为 MIDI 轨道。

    Args:
        notes: 音符和时间的元组列表，格式为 [(note_1, time_1), (note_2, time_2), ...]

    Returns:
        包含音符开启、关闭事件及结束标记的 MIDI 轨道。
    """
    # 生成 MIDI 事件
    events = []
    now = 0
    for note, time in notes:
        now += time * TIME_PRECISION
        events.append(["note_on", note, now])  # 音符开启事件
        events.append(["note_off", note, now + TIME_PRECISION])  # 音符关闭事件

    events.sort(key=lambda event: event[2])  # 按开始时间排序

    # 创建 MIDI 轨道
    track = []
    now = 0  # 初始化当前时间

    # 创建 MIDI 消息
    for note_type, note, start_at in events:
        track.append(mido.Message(
            note_type,
            channel=0,  # 音符通道
            note=note,
            velocity=100 if note_type == "note_on" else 0,  # 音符响度
            time=start_at - now  # 时间间隔
        ))
        now = start_at  # 更新当前时间

    # 返回 MIDI 轨道，添加结束事件
    return mido.MidiTrack(track + [mido.MetaMessage("end_of_track")])  # 结束标记


def generate_midi(
    prompt: list[tuple[int, int]],
    model: MidiNet,
    seed: int,
    length: int,
    temperature: float = 1,
    top_p: float = 0.8,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0
) -> mido.MidiTrack:
    """
    生成 MIDI 音乐轨道。

    Args:
        model: 使用的模型
        prompt: 预处理提示，格式为 [(note_1, time_1), (note_2, time_2), ...]
        seed: 随机种子，确保结果一致
        length: 生成音符序列的长度
        temperature: 控制生成多样性，低温度结果更确定，高温度增加多样性
        top_p: 核采样的累积概率阈值
        frequency_penalty: 频率惩罚强度
        presence_penalty: 出现惩罚强度

    Returns:
        包含 MIDI 消息的轨道。
    """
    model.eval()  # 将模型设置为评估模式
    generator = torch.Generator().manual_seed(seed)  # 固定随机种子，确保每次生成结果一致
    input_prompt = [note * NOTE_DURATION_COUNT + time for note, time in prompt]  # 记录已生成的音符

    for i in tqdm.tqdm(range(length), desc="Generate MIDI"):
        # 将已生成的音符和时间输入模型进行预测
        model_input = torch.tensor(input_prompt, dtype=torch.long).unsqueeze(0)

        # 获取模型预测
        with torch.no_grad():
            prediction = model(model_input)[0, -1, :]

        # 将预测值按温度调整，控制分布的平滑度
        prediction = F.softmax(prediction / temperature, dim=-1)

        # 应用 Frequency Penalty
        if frequency_penalty > 0.0:
            # 计算每个音符的频率
            note_counts = torch.bincount(torch.tensor(input_prompt), minlength=prediction.size(-1))
            frequency_factor = 1 - (note_counts / note_counts.sum()).clamp(max=1) * frequency_penalty
            prediction *= frequency_factor

        # 应用 Presence Penalty
        if presence_penalty > 0.0:
            unique_notes = set(input_prompt)
            presence_factor = torch.ones_like(prediction)
            for note in unique_notes:
                presence_factor[note] *= (1 - presence_penalty)
            prediction *= presence_factor

        # 确保概率不为负
        prediction = F.relu(prediction)

        # 重新归一化
        prediction /= prediction.sum()

        # 根据累积概率选择保留的候选值
        sorted_probs, sorted_indices = torch.sort(prediction, descending=True, dim=-1)  # 按概率排序
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # 计算累积概率
        sorted_indices_to_keep = cumulative_probs <= top_p  # 选择累积概率小于top_p的部分
        sorted_indices_to_keep[..., -1] = 1  # 确保最后一个token被保留
        sorted_probs_to_keep = sorted_probs * sorted_indices_to_keep  # 过滤后的概率
        sorted_probs_to_keep = sorted_probs_to_keep / sorted_probs_to_keep.sum(dim=-1, keepdim=True)  # 重新归一化

        # 从候选的概率分布中进行采样
        token = torch.multinomial(sorted_probs_to_keep, 1, generator=generator)  # 从概率分布中采样
        token = torch.gather(sorted_indices, -1, token)  # 映射回原始索引空间

        # 获取生成的音符和时间
        input_prompt.append(token.item())

    # 将生成的音符和时间配对
    notes = list(map(lambda note: (note // NOTE_DURATION_COUNT, note % NOTE_DURATION_COUNT), input_prompt))
    return model_output_to_track(notes)  # 转化为 MIDI 轨道


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
        length=128)
    ]).save("example.mid")


if __name__ == "__main__":
    main()
