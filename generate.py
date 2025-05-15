"生成MIDI文件"
# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import argparse
import pathlib
import random
import mido
import math
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from typing import Iterator, Generator, Optional
from collections import deque

# 内置音乐曲库（爱情买卖）
EXAMPLE_MIDI = [(45, 0), (76, 0), (52, 1), (57, 1), (81, 0), (59, 1), (60, 1), (84, 0), (64, 1), (84, 1), (76, 2), (81, 2), (57, 1), (69, 1), (84, 0), (64, 1), (60, 1), (43, 2), (83, 0), (50, 1), (55, 1), (83, 0), (57, 1), (59, 1), (81, 0), (62, 1), (79, 1), (76, 2), (55, 3), (67, 1), (62, 1), (59, 1), (55, 1), (41, 1), (74, 0), (48, 1), (53, 1), (74, 0), (55, 1), (57, 1), (74, 0), (60, 1), (69, 1), (74, 2), (76, 2), (53, 1), (65, 1), (79, 0), (60, 1), (57, 1), (43, 2), (76, 0), (50, 1), (55, 1), (83, 0), (57, 1), (59, 1), (83, 0), (62, 1), (79, 1), (76, 2), (74, 3), (71, 1), (67, 1), (62, 1), (64, 1), (45, 1), (76, 0), (52, 1), (57, 1), (81, 0), (59, 1), (60, 1), (84, 0), (64, 1), (84, 1), (76, 2), (81, 2), (57, 1), (69, 1), (84, 0), (64, 1), (60, 1), (43, 2), (88, 0), (50, 1), (55, 1), (88, 0), (57, 1), (59, 1), (86, 0), (62, 1), (84, 1), (86, 2), (55, 3), (67, 1), (62, 1), (59, 1), (55, 1), (41, 1), (88, 0), (48, 1), (53, 1), (55, 1), (88, 0), (57, 1), (86, 0), (60, 1), (84, 1), (43, 2), (86, 0), (50, 1), (55, 1), (57, 1), (86, 0), (59, 1), (84, 0), (62, 1), (83, 1), (45, 2), (79, 0), (52, 1), (57, 1), (76, 0), (59, 1), (60, 1), (79, 0), (64, 1), (81, 1), (81, 2), (76, 3), (72, 1), (69, 1), (64, 1), (69, 1)]


# 在非 Jupyter 环境下导入常量、模型、检查点、工具、分词库
if "get_ipython" not in globals():
    from constants import TIME_PRECISION, DEFAULT_DIM_HEAD, DEFAULT_NUM_HEADS, KEY_UP, KEY_DOWN, OCTAVE_JUMP_UP, OCTAVE_JUMP_DOWN, LOOKAHEAD_COUNT
    from model import MidiNet
    from checkpoint import load_checkpoint
    from utils import midi_to_notes, notes_to_sheet, sheet_to_notes
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

    for pitch, interval in notes:
        # 计算事件绝对时间
        cumulative_time += interval * TIME_PRECISION

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


def generate_sheet(
    prompt: str,
    model: MidiNet,
    tokenizer: PreTrainedTokenizerFast,
    seed: int,
    temperature: float,
    top_k: Optional[int],
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
        # 模型前向计算
        with torch.no_grad():
            # 增加输入张量的批次维度，再进行推理
            logits = model(input_tensor.unsqueeze(0), torch.zeros(1, *input_tensor.size()))[0, -1, :]

        # 屏蔽特殊标记(BOS/PAD/UNK)
        logits[tokenizer.bos_token_id] = -torch.inf
        logits[tokenizer.pad_token_id] = -torch.inf
        logits[tokenizer.unk_token_id] = -torch.inf

        # Top-K
        if top_k:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[-1]] = -torch.inf

        # 应用温度参数并计算概率分布
        probs = F.softmax(logits / temperature, dim=-1)

        # 如果存在需要调整概率的事件，则降低其相关token的概率
        if events_to_dampen:
            for event, frequency_reduction in events_to_dampen:
                for token, token_id in tokenizer.vocab.items():
                    if event in token and token != tokenizer.eos_token:
                        probs[token_id] *= 1 - frequency_reduction

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
    top_k: Optional[float] = None,
    max_pitch_span_semitones: int = 64,
    max_length: Optional[int] = None,
    device: torch.device = None
) -> Iterator[tuple[int, int]]:
    """
    实时流式生成MIDI音符序列。

    该函数基于输入音符提示序列，采用后台线程异步生成MIDI事件流，同时主线程实时输出音符(pitch, interval)。
    音高跨度超过设定阈值时，会动态降低升高/降低音调事件的概率，以避免音域漂移。

    处理流程：
    1. 将输入的(pitch, interval)音符序列转换为模型可识别的乐谱事件
    2. 启动后台线程异步调用生成器生成事件序列
    3. 主线程实时读取事件流，转换为音符并输出
    4. 根据当前音高跨度实时调整音调变换事件的生成概率

    Args:
        prompt: 初始音符序列，每个元素为(音高, 间隔时间)元组
        model: 预训练的音乐生成模型实例
        tokenizer: 乐谱事件与文本互相转换的分词器
        seed: 随机种子，None表示随机生成
        temperature: 控制生成多样性的温度参数(默认1.0)
        top_k: 仅对概率前`top_k`的token采样，减小随机性
        max_pitch_span_semitones: 音高跨度超过该值时将进行音高调整（单位: 半音）
        max_length: 限制最多生成的音符数量（计算长度不包括提示音符）
        device: 模型运行的设备(cpu/cuda)

    Yields:
        生成的音符(音高pitch, 间隔时间interval)

    Note:
        1. 采用守护线程确保主程序退出时自动终止生成
        2. 当音高跨度超过阈值时自动降低升调/降调事件概率
        3. 音高调整强度与超出阈值幅度成正比

    Examples:
        >>> # 获取 prompt 并开始生成
        >>> prompt = midi_to_notes(mido.MidiFile("Touhou Broken_Moon.mid"))
        >>> for pitch, interval in generate_midi(prompt, model, tokenizer, seed=19890604):
        ...     print(f"生成音符: 音高{pitch}，与前一个音符的时间间隔{interval}")
    """
    # 初始化随机种子
    if seed is None:
        seed = random.randint(0, 2 ** 32)

    # 转换输入音符为乐谱格式
    sheet_music, _ = notes_to_sheet(prompt)
    prompt_text = data_to_str(sheet_music)

    # 初始化滑动窗口，用于追踪最近生成的音高
    recent_pitch_window = deque([0])

    # 初始化生成音符的计数器
    generated_notes_counter = 0

    # 创建主生成器
    generator = generate_sheet(prompt_text, model, tokenizer, seed, temperature, top_k, device)

    # 初始化状态变量
    global_offset = sheet_music.count(KEY_UP) - sheet_music.count(KEY_DOWN)  # 当前全局偏移
    octave_offset = 0  # 当前八度偏移
    accumulated_interval = 0  # 累计时间间隔

    # 返回提示音符
    for note in sheet_to_notes(sheet_music):
        yield note

    # 生成token并转化为音符
    for token in generator:
        for event in str_to_data(token):  # 加入提示音符
            if event < 12:
                # 计算并生成最终音符
                final_pitch = event - global_offset + octave_offset * 12

                # 返回音符
                yield final_pitch, accumulated_interval

                # 判断是否终止生成循环
                generated_notes_counter += 1
                if max_length is not None and generated_notes_counter >= max_length:
                    return

                # 更新窗口
                recent_pitch_window.append(final_pitch)

                # 控制窗口大小
                if len(recent_pitch_window) > LOOKAHEAD_COUNT:
                    recent_pitch_window.popleft()

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

        # 计算当前音高跨度
        pitch_span = max(recent_pitch_window) - min(recent_pitch_window)

        # 检查是否需要调整
        if pitch_span > max_pitch_span_semitones:
            # 每超过1个八度（12半音）增加10%惩罚
            suppression_ratio = (pitch_span - max_pitch_span_semitones) / 12 * 0.1

            # 对升调/降调应用概率抑制
            if global_offset > 0:
                suppressed_events = [KEY_UP, OCTAVE_JUMP_UP]
            else:
                suppressed_events = [KEY_DOWN, OCTAVE_JUMP_DOWN]

            try:
                generator.send([(event, suppression_ratio) for event in data_to_str(suppressed_events)])
                pass
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
    parser.add_argument("-m", "--midi-path", type=pathlib.Path, help="指定的 MIDI 文件，将作为生成的音乐的前面部分。如果未指定，将使用内置的音乐来生成。")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="采样温度参数，值越高生成结果越多样，值越低结果越保守")
    parser.add_argument("-k", "--top-k", type=float, default=None, help="仅对概率前`top_k`个token采样，减小随机性")
    parser.add_argument("-s", "--seed", type=int, help="随机种子，不指定表示随机生成")
    parser.add_argument("-p", "--max-pitch-span-semitones", type=int, default=64, help="触发音高调整的阈值（半音数），当生成的音高跨度大于阈值时，包含音调上升或下降事件的 token 将被降低概率。")
    parser.add_argument("-l", "--max-length", type=int, help="限制最多生成的音符数量。如果不指定，将会持续生成直到遇到结束标志。")
    parser.add_argument("-n", "--num-heads", type=int, default=DEFAULT_NUM_HEADS, help="模型注意力头的数量。")
    args = parser.parse_args()

    # 加载模型的预训练检查点
    tokenizer, state_dict = load_checkpoint(args.ckpt_path, train=False)

    # 加载音乐生成的提示部分
    prompt_notes = EXAMPLE_MIDI.copy()
    if args.midi_path:
        try:
            prompt_notes = midi_to_notes(mido.MidiFile(args.midi_path))
        except Exception as e:
            print(f"加载指定的 MIDI 文件时出错: {e}\n将选择内置的音乐作为代替。")

    # 推导模型参数
    vocab_size, dim_model = state_dict["embedding.weight"].size()
    dim_feedforward = state_dict["layers.0.feedforward.0.weight"].size(0)
    num_layers = len(set(key.split(".")[1] for key in state_dict.keys() if key.startswith("layers.")))
    if dim_model % args.num_heads == 0:
        num_heads = args.num_heads
    elif dim_model % DEFAULT_DIM_HEAD == 0:
        num_heads = dim_model // DEFAULT_DIM_HEAD
    elif dim_model % DEFAULT_NUM_HEADS == 0:
        num_heads = DEFAULT_NUM_HEADS
    dim_head = dim_model // num_heads

    # 打印模型参数
    print(f"模型参数:\n- 词汇表大小: {vocab_size}\n- 嵌入层维度: {dim_model}\n- 前馈层维度: {dim_feedforward}\n- 注意力头的数量: {num_heads}\n- 层数: {num_layers}\n")

    # 初始化模型并加载状态
    model = MidiNet(vocab_size, num_heads, dim_head, dim_feedforward, num_layers)
    model.load_state_dict(state_dict)

    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转移模型到设备并设置为评估模式
    model = model.to(device).eval()

    # 模型推理生成
    music = []
    for note in generate_midi(prompt_notes, model, tokenizer, seed=args.seed, temperature=args.temperature, max_pitch_span_semitones=args.max_pitch_span_semitones, max_length=args.max_length, device=device):
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
