"这个模块实现了 MidiNet 模型的生成和处理功能，包括音乐生成、音符转换等。"

# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
import random
import math
from typing import Iterator, Generator, Optional
import mido
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from utils.checkpoint import load_checkpoint, extract_config
from utils.model import MidiNet
from utils.constants import KEY_UP, KEY_DOWN, OCTAVE_JUMP_UP, OCTAVE_JUMP_DOWN, LOOKAHEAD_COUNT
from utils.midi import midi_to_notes, notes_to_sheet, sheet_to_notes, notes_to_track
from tokenizer import data_to_str, str_to_data


@torch.inference_mode()
def generate_sheet(
    prompt: str,
    model: MidiNet,
    tokenizer: PreTrainedTokenizerFast,
    seed: int,
    temperature: float,
    top_k: Optional[int],
    repetition_penalty: float,
    device: torch.device
) -> Generator[str, Optional[list[tuple[str, float]]], None]:
    """
    使用自回归方式生成音乐乐谱事件序列的生成器函数。

    本函数通过预训练的音乐生成模型，以给定的提示文本为起点，逐步生成音乐乐谱事件序列。
    生成过程支持通过外部交互动态调整特定事件的生成概率，并可通过随机种子和温度参数控制生成效果。

    Args:
        prompt: 用于初始化生成的乐谱事件序列文本，应符合tokenizer的编码格式
        model: 预训练的音乐生成模型，应实现类似语言模型的接口
        tokenizer: 用于乐谱事件与token相互转换的分词器实例
        seed: 随机数生成种子，用于控制生成过程的确定性
        temperature: 采样温度参数，值越高生成结果越多样，值越低结果越保守
        top_k: 仅对概率前`top_k`个token采样，减小随机性
        repetition_penalty: 重复惩罚，大于 1 则减少重复
        device: 指定模型运行的计算设备

    Yields:
        每次迭代生成一个乐谱事件token的字符串表示

    Receives:
        通过send()方法接收的调整指令格式为list[tuple[事件, 概率衰减值]]，用于降低特定事件的生成概率

    Note:
        1. 生成过程将持续直到产生EOS标记
        2. 可通过生成器的send()方法实时调整特定事件的生成概率
        3. 温度参数建议范围(0.1, 1.0)，极端值可能导致生成质量下降
        4. 降低特定事件的生成概率时如果有 token 包含若干个被指定的事件，那么它会被降低不止一次概率

    Examples:
        >>> # 转换 prompt 并开始生成
        >>> prompt = data_to_str(notes_to_sheet(midi_to_notes(mido.MidiFile("Touhou Broken_Moon.mid"))))
        >>> generator = generate_sheet(prompt, model, tokenizer, seed=1989, temperature=0.604)
        >>> for token in generator:
        ...     generator.send([(data_to_str([KEY_UP]), 0.1), (data_to_str([KEY_DOWN]), 0.1)])  # 减小 KEY_UP 和 KEY_DOWN 事件出现的概率
        ...     prompt += token
        >>> # 转换为 MIDI 轨道并保存
        >>> generated_notes = sheet_to_notes(str_to_data(prompt))
        >>> mido.MidiFile(trakcs=[notes_to_track(generated_notes)]).save("generated.mid")
    """
    # 初始化随机数生成器并设置种子
    generator = torch.Generator(device=device).manual_seed(seed)

    # 编码提示文本并移除结束标记
    input_tensor = torch.tensor(tokenizer.encode(prompt)[:-1], device=device)

    # 用于存储需要调整概率的事件及其频率衰减值
    events_to_dampen = None

    # 自回归生成循环
    while True:
        # 增加输入张量的批次维度，再进行推理
        logits = model(input_tensor.unsqueeze(0))[0, -1, :]

        # 屏蔽特殊标记(BOS/PAD/UNK)
        logits[tokenizer.bos_token_id] = -torch.inf
        logits[tokenizer.pad_token_id] = -torch.inf
        logits[tokenizer.unk_token_id] = -torch.inf

        # Repetition Penalty
        score = torch.gather(logits, 0, input_tensor)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(0, input_tensor, score)

        # Top-K
        if top_k:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[-1]] = -torch.inf

        # 应用温度参数并计算概率分布
        probs = F.softmax(logits / temperature, dim=-1)
        probs = torch.nan_to_num(probs, nan=-torch.inf)

        # 如果存在需要调整概率的事件，则降低其相关token的概率
        if events_to_dampen is not None:
            for event, frequency_reduction in events_to_dampen:
                for token, token_id in tokenizer.vocab.items():
                    if event in token and token != tokenizer.eos_token:
                        probs[token_id] *= (1 - frequency_reduction) ** token.count(event)

        # 保证概率不为负并重新归一化
        probs = F.relu(probs)
        probs = probs / probs.sum()

        # 从概率分布中采样下一个标记
        next_token = torch.multinomial(probs, 1, generator=generator).item()

        # 遇到结束标记则停止生成
        if next_token == tokenizer.eos_token_id:
            break

        # 返回生成的标记并获取要求减少频率的事件及要求减少的概率
        events_to_dampen = yield tokenizer.convert_ids_to_tokens(next_token)

        # 将新标记添加到输入中用于下一次迭代
        input_tensor = torch.cat([input_tensor, torch.tensor([next_token], device=device)], dim=-1)


def generate_midi(
    prompt: list[tuple[int, int]],
    model: MidiNet,
    tokenizer: PreTrainedTokenizerFast,
    seed: Optional[int] = None,
    temperature: float = 1.,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.2,
    max_pitch_span_semitones: float = 20.,
    max_length: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Iterator[tuple[int, int]]:
    """
    MIDI音乐流式生成器

    通过神经网络模型实时生成音乐音符序列，支持动态音高稳定性控制

    参数:
        prompt: 初始音符提示序列，每个元素为(音高, 间隔时间)的元组
        model: 用于生成的神经网络模型
        tokenizer: 文本tokenizer
        seed: 随机种子，None表示随机生成
        temperature: 控制生成随机性的温度参数
        top_k: 只考虑概率最高的k个token
        repetition_penalty: 重复惩罚系数
        pitch_volatility_threshold: 音高波动阈值(半音标准差)，超过此值会触发抑制机制
        max_length: 生成的最大音符数量，若为 None，则会持续生成直到遇到结束标志
        device: 使用的计算设备(CPU/GPU)

    返回:
        生成器，产出(音高, 间隔时间)元组
    """
    # 初始化随机种子
    if seed is None:
        seed = random.randint(0, 2 ** 32)

    # 设备判定与赋值
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转换输入音符为乐谱格式
    if prompt:
        sheet_music, _ = notes_to_sheet(prompt)
        prompt_text = data_to_str(sheet_music)
    else:
        sheet_music = []
        prompt_text = ""

    # 初始化窗口，用于追踪生成的音高
    pitch_window = np.array([], dtype=int)

    # 创建主生成器
    generator = generate_sheet(prompt_text, model, tokenizer, seed, temperature, top_k, repetition_penalty, device)

    # 初始化状态变量
    global_offset = sheet_music.count(KEY_UP) - sheet_music.count(KEY_DOWN)  # 当前全局偏移
    octave_offset = 0  # 当前八度偏移
    accumulated_interval = 0  # 累计时间间隔

    # 返回提示音符
    for pitch, interval in sheet_to_notes(sheet_music):
        yield pitch, interval
        pitch_window = np.concatenate((pitch_window, [pitch]))

    # 生成token并转化为音符
    for token in generator:
        for event in str_to_data(token):  # 加入提示音符
            if event < 12:
                # 计算并生成最终音符
                final_pitch = event - global_offset + octave_offset * 12

                # 返回音符
                yield final_pitch, accumulated_interval

                # 更新窗口和全局平均音高
                pitch_window = np.concatenate((pitch_window, [final_pitch]))

                # 判断是否终止生成循环
                if max_length is not None and len(pitch_window) >= max_length:
                    return

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

        # 如果音高波动超过阈值则进行调整
        if pitch_window.std() > max_pitch_span_semitones:
            # 计算抑制比率(基于当前波动与阈值的差值)
            suppression_ratio = (pitch_window.std() - max_pitch_span_semitones) * 0.1

            # 根据最近音高趋势决定抑制方向
            if pitch_window[-LOOKAHEAD_COUNT:].mean() > pitch_window.mean():
                suppressed_events = [KEY_UP, OCTAVE_JUMP_UP]
            else:
                suppressed_events = [KEY_DOWN, OCTAVE_JUMP_DOWN]

            # 向生成器发送抑制信号
            try:
                generator.send([(event, suppression_ratio) for event in data_to_str(suppressed_events)])
            except StopIteration:
                pass


def center_pitches(pitches: list[int]) -> list[tuple[int, int]]:
    """
    将音符序列的音高居中化处理，使平均音高移动到64附近。

    工作流程:
    1. 计算当前平均音高
    2. 调整所有音高使平均音高移动到目标值(64)

    Args:
        pitches: 音高序列

    Returns:
        处理后的音高序列，音高整体平移

    Examples:
        >>> notes = [(60, 4), (64, 2)]  # 平均音高62
        >>> pitches, intervals = zip(*notes)
        >>> list(zip(center_pitches(pitches), intervals))
        [(62, 4), (66, 2)]  # 平均音高变为64
    """
    # 空序列返回
    if not pitches:
        return []

    # 计算当前平均音高（使用浮点除法）
    current_avg = sum(pitches) / len(pitches)

    # 计算需要平移的半音数（目标音高64）
    pitch_shift = round(64 - current_avg)

    # 平移所有音高并重新组合（使用列表推导式更高效）
    return [pitch + pitch_shift for pitch in pitches]


def clamp_midi_pitch(pitches: list[int]):
    """
    将音符音高值标准化到0-127的有效MIDI音高范围内。
    对于超出范围的音高，通过加减12的整数倍（八度）将其调整到有效范围内。

    工作流程:
    1. 遍历输入的音高列表
    2. 对于每个音高:
       - 如果大于127，减去适当的12的倍数使其≤127
       - 如果小于0，加上适当的12的倍数使其≥0
    3. 返回标准化后的音高列表

    Args:
        pitches: 原始音高列表，可能包含超出MIDI范围 [0, 127] 的值

    Returns:
        标准化后的音高列表，所有值都在 [0, 127] 范围内

    Examples:
        >>> clamp_midi_pitch([128, -1, 60, 255])
        [116, 11, 60, 123]
        >>> clamp_midi_pitch([-13, 140])
        [11, 116]
    """
    normalized_pitches = []
    for pitch in pitches:
        if pitch > 127:
            # 计算需要减去多少个八度(12的倍数)才能不超过127
            octaves_to_subtract = math.ceil((pitch - 127) / 12)
            pitch -= octaves_to_subtract * 12
        elif pitch < 0:
            # 计算需要加上多少个八度(12的倍数)才能不小于0
            octaves_to_add = math.floor(pitch / 12)
            pitch -= octaves_to_add * 12
        normalized_pitches.append(pitch)
    return normalized_pitches


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="以指定 MIDI 为前面部分并生成音乐和保存。")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点的路径")
    parser.add_argument("output_path", type=pathlib.Path, help="MIDI 文件保存路径。生成的 MIDI 文件将会保存到这里。")
    parser.add_argument("-m", "--midi-path", type=pathlib.Path, help="指定的 MIDI 文件，将作为生成的音乐的前面部分。如果未指定，将从头开始生成。")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="采样温度参数，值越高生成结果越多样，值越低结果越保守，默认为 %(default)s")
    parser.add_argument("-k", "--top-k", type=int, help="仅对概率前`top_k`个token采样，减小随机性")
    parser.add_argument("-r", "--repetition-penalty", type=float, default=1.2, help="重复惩罚，大于 1 则减少重复，默认为 %(default)s")
    parser.add_argument("-s", "--seed", type=int, help="随机种子，不指定表示随机生成")
    parser.add_argument("-p", "--max-pitch-span-semitones", type=int, default=64, help="触发音高调整的阈值（半音数），当生成的音高跨度大于阈值时，包含音调上升或下降事件的 token 将被降低概率。默认为 %(default)s")
    parser.add_argument("-l", "--max-length", type=int, help="限制最多生成的音符数量。如果不指定，将会持续生成直到遇到结束标志。")
    args = parser.parse_args()

    # 加载模型的预训练检查点
    tokenizer, state_dict = load_checkpoint(args.ckpt_path)

    # 加载音乐生成的提示部分
    prompt_notes = []  # 默认空提示音符列表
    if args.midi_path:
        try:
            prompt_notes = midi_to_notes(mido.MidiFile(args.midi_path))
        except Exception as e:  # 捕获加载 MIDI 文件时的异常
            print(f"加载指定的 MIDI 文件时出错: {e}\n将选择内置的音乐作为代替。")

    # 推导模型参数
    config = extract_config(state_dict)

    # 打印模型参数
    print(f"模型参数:\n- 词汇表大小: {config.dim_head * config.num_heads}\n- 注意力头数: {config.num_heads}\n- 注意力头的维度: {config.dim_head}\n- 前馈层维度: {config.dim_feedforward}\n- 层数: {config.num_layers}\n")

    # 初始化模型并加载状态
    model = MidiNet(config)
    model.load_state_dict(state_dict)

    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转移模型到设备并设置为评估模式
    model = model.to(device).eval()

    # 模型推理生成
    music = []
    for note in generate_midi(prompt_notes, model, tokenizer, seed=args.seed, temperature=args.temperature, repetition_penalty=args.repetition_penalty, max_pitch_span_semitones=args.max_pitch_span_semitones, max_length=args.max_length, device=device):
        music.append(note)
        print(note)

    # 使音高居中
    pitches, intervals = zip(*music)
    pitches = center_pitches(pitches)

    # 音高上移、下移，以满足所有音高在 [0, 127] 范围内
    pitches = clamp_midi_pitch(pitches)

    # 再次居中音高
    pitches = center_pitches(pitches)

    # 重组为音符序列
    music = zip(pitches, intervals)

    # 转换为 MIDI 轨道并保存为文件
    track = notes_to_track(music)
    mido.MidiFile(tracks=[track]).save(args.output_path)


if __name__ == "__main__":
    main()
