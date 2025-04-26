"生成MIDI文件"
# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import sys
import pathlib
import mido
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from transformers import PreTrainedTokenizerFast

# 内置音乐曲库
LOVE_TRADING_MIDI = [(45, 0), (76, 0), (52, 1), (57, 1), (81, 0), (59, 1), (60, 1), (84, 0), (64, 1), (84, 1), (76, 2), (81, 2), (57, 1), (69, 1), (84, 0), (64, 1), (60, 1), (43, 2), (83, 0), (50, 1), (55, 1), (83, 0), (57, 1), (59, 1), (81, 0), (62, 1), (79, 1), (76, 2), (55, 3), (67, 1), (62, 1), (59, 1), (55, 1), (41, 1), (74, 0), (48, 1), (53, 1), (74, 0), (55, 1), (57, 1), (74, 0), (60, 1), (69, 1), (74, 2), (76, 2), (53, 1), (65, 1), (79, 0), (60, 1), (57, 1), (43, 2), (76, 0), (50, 1), (55, 1), (83, 0), (57, 1), (59, 1), (83, 0), (62, 1), (79, 1), (76, 2), (74, 3), (71, 1), (67, 1), (62, 1), (64, 1), (45, 1), (76, 0), (52, 1), (57, 1), (81, 0), (59, 1), (60, 1), (84, 0), (64, 1), (84, 1), (76, 2), (81, 2), (57, 1), (69, 1), (84, 0), (64, 1), (60, 1), (43, 2), (88, 0), (50, 1), (55, 1), (88, 0), (57, 1), (59, 1), (86, 0), (62, 1), (84, 1), (86, 2), (55, 3), (67, 1), (62, 1), (59, 1), (55, 1), (41, 1), (88, 0), (48, 1), (53, 1), (55, 1), (88, 0), (57, 1), (86, 0), (60, 1), (84, 1), (43, 2), (86, 0), (50, 1), (55, 1), (57, 1), (86, 0), (59, 1), (84, 0), (62, 1), (83, 1), (45, 2), (79, 0), (52, 1), (57, 1), (76, 0), (59, 1), (60, 1), (79, 0), (64, 1), (81, 1), (81, 2), (76, 3), (72, 1), (69, 1), (64, 1), (69, 1)]

# 根据是否在 Jupyter 环境下导入不同库
if "get_ipython" in globals():
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
    from constants import TIME_PRECISION
    from model import MidiNet
    from checkpoint import load_checkpoint
    from utils import notes_to_sheet, sheet_to_notes
    from tokenizer import data_to_str, str_to_data


def notes_to_track(notes: list[int]) -> mido.MidiTrack:
    """
    将音符和时间信息转换为 MIDI 轨道。

    Args:
        notes: 音符间隔格式的列表

    Returns:
        包含音符事件及结束标记的 MIDI 轨道
    """
    # 生成 MIDI 事件队列
    events = []
    cumulative_time = 0  # 累计时间

    for pitch, time in notes:
        # 计算事件绝对时间
        cumulative_time += time * TIME_PRECISION

        # 添加音符开启和关闭事件
        events.append(("note_on", pitch, cumulative_time))
        events.append(("note_off", pitch, cumulative_time + TIME_PRECISION))

    # 按事件发生时间排序（确保事件顺序正确）
    events.sort(key=lambda x: x[2])

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


def generate_midi(
    prompt: list[tuple[int, int]],
    model: MidiNet,
    tokenizer: PreTrainedTokenizerFast,
    seed: int,
    temperature: float = 1,
    show_progress: bool = True,
    device: torch.device = None
) -> mido.MidiTrack:
    """
    使用模型生成 MIDI 音乐轨道

    Args:
        model: 用于生成的神经网络模型
        prompt: 初始音符序列，格式为 [(音高, 时间单位), ...]
        tokenizer: 分词器
        seed: 随机种子
        temperature: 采样温度（大于1增加多样性，小于1减少随机性）
        show_progress: 显示进度条
        device: 设备

    Returns:
        生成的 MIDI 轨道
    """
    # 将音符和时间转换为电子乐谱
    sheet, _ = notes_to_sheet(prompt)

    # 将电子乐谱通过分词器转化为模型输入序列
    input_prompt = tokenizer.encode(data_to_str(sheet))[:-1]

    model = model.to(device)  # 移动模型到设备上
    model.eval()  # 将模型设置为评估模式
    generator = torch.Generator(device=device).manual_seed(seed)  # 固定随机种子，确保每次生成结果一致

    # 生成循环
    if show_progress:
        progress_bar = tqdm(desc="Generate MIDI")

    while input_prompt[-1] != tokenizer.eos_token_id:
        # 准备模型输入
        input_tensor = torch.tensor(input_prompt, dtype=torch.long).unsqueeze(0)

        # 获取模型预测
        with torch.no_grad():
            logits = model(input_tensor)
            logits = logits[0, -1, :]

        # 应用温度缩放
        probs = F.softmax(logits / temperature, dim=-1)

        # 采样
        input_prompt.append(torch.multinomial(probs, 1, generator=generator).item())

        # 更新进度条
        progress_bar.update()

    progress_bar.close()

    # 模型输出转换为音符时间
    output_notes = sheet_to_notes(str_to_data("".join(tokenizer.convert_ids_to_tokens(input_prompt[1:-1]))))

    # 调整到音高平均值到中间
    pitches = list(zip(*output_notes))[0]
    offset = 64 - int(sum(pitches) / len(pitches))
    output_notes = [(pitch + offset, interval) for pitch, interval in output_notes]

    try:
        return notes_to_track(output_notes)  # 转化为 MIDI 轨道
    except ValueError as e:
        print("音符:", output_notes, file=sys.stderr)
        print("转化音符到轨道时发生错误（最高音高与最低音高差距大于128）:", e, file=sys.stderr)


def main():
    # 加载模型的预训练检查点
    tokenizer, state = load_checkpoint(pathlib.Path("ckpt"), train=False)

    # 获取模型参数
    vocab_size, d_model = state["embedding.weight"].size()
    dim_feedforward = state["transformer.layers.0.linear1.weight"].size(0)
    num_heads = 8
    num_layers = len(set(key.split(".")[2] for key in state.keys() if key.startswith("transformer.layers.")))

    # 初始化模型并加载状态
    model = MidiNet(vocab_size, d_model, num_heads, dim_feedforward, num_layers)
    model.load_state_dict(state)

    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转移模型到设备
    model = model.to(device)

    # 检查是否使用多GPU
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # 使用 DataParallel 进行多 GPU 推理

    track = generate_midi(LOVE_TRADING_MIDI, model, tokenizer, seed=42, device=device)
    if track is not None:
        mido.MidiFile(tracks=[track]).save("example.mid")


if __name__ == "__main__":
    main()
