"生成MIDI文件"
# Copyright (C)  thiliapr 2024-2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import threading
import pathlib
import mido
import time
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from transformers import PreTrainedTokenizerFast
from typing import Iterator, Iterable
from collections import deque

# 内置音乐曲库
LOVE_TRADING_MIDI = [(45, 0), (76, 0), (52, 1), (57, 1), (81, 0), (59, 1), (60, 1), (84, 0), (64, 1), (84, 1), (76, 2), (81, 2), (57, 1), (69, 1), (84, 0), (64, 1), (60, 1), (43, 2), (83, 0), (50, 1), (55, 1), (83, 0), (57, 1), (59, 1), (81, 0), (62, 1), (79, 1), (76, 2), (55, 3), (67, 1), (62, 1), (59, 1), (55, 1), (41, 1), (74, 0), (48, 1), (53, 1), (74, 0), (55, 1), (57, 1), (74, 0), (60, 1), (69, 1), (74, 2), (76, 2), (53, 1), (65, 1), (79, 0), (60, 1), (57, 1), (43, 2), (76, 0), (50, 1), (55, 1), (83, 0), (57, 1), (59, 1), (83, 0), (62, 1), (79, 1), (76, 2), (74, 3), (71, 1), (67, 1), (62, 1), (64, 1), (45, 1), (76, 0), (52, 1), (57, 1), (81, 0), (59, 1), (60, 1), (84, 0), (64, 1), (84, 1), (76, 2), (81, 2), (57, 1), (69, 1), (84, 0), (64, 1), (60, 1), (43, 2), (88, 0), (50, 1), (55, 1), (88, 0), (57, 1), (59, 1), (86, 0), (62, 1), (84, 1), (86, 2), (55, 3), (67, 1), (62, 1), (59, 1), (55, 1), (41, 1), (88, 0), (48, 1), (53, 1), (55, 1), (88, 0), (57, 1), (86, 0), (60, 1), (84, 1), (43, 2), (86, 0), (50, 1), (55, 1), (57, 1), (86, 0), (59, 1), (84, 0), (62, 1), (83, 1), (45, 2), (79, 0), (52, 1), (57, 1), (76, 0), (59, 1), (60, 1), (79, 0), (64, 1), (81, 1), (81, 2), (76, 3), (72, 1), (69, 1), (64, 1), (69, 1)]

# 在非 Jupyter 环境下导入常量、模型、检查点、工具、分词库
if "get_ipython" not in globals():
    from constants import TIME_PRECISION
    from model import MidiNet
    from checkpoint import load_checkpoint
    from utils import notes_to_sheet, sheet_to_notes
    from tokenizer import data_to_str, str_to_data


class BufferStream(Iterable):
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

    def send(self, data: list[int]):
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

    def __iter__(self):
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
    device: torch.device
) -> Iterator[str]:
    """
    根据给定的提示文本生成音乐乐谱事件序列。

    使用预训练的语言模型以自回归方式生成音乐事件序列，直到遇到结束标记或达到最大长度。
    生成过程可以通过随机种子和温度参数进行控制。

    Args:
        prompt: 用于生成乐谱的起始文本提示
        model: 用于音乐生成的预训练模型
        tokenizer: 乐谱事件的分词器
        seed: 控制生成随机性的种子值
        temperature: 控制生成多样性的温度参数
        device: 模型运行的计算设备

    Returns:
        一个生成器，每次迭代返回一个生成的乐谱事件

    Examples:
        >>> import mido
        >>> from utils import midi_to_notes, notes_to_sheet
        >>> from tokenizer import data_to_str, str_to_data
        >>> model = MidiNet(...)
        >>> model.load_state_dict(torch.load("ckpt/model.pth"))
        >>> tokenizer = PreTrainedTokenizerFast.from_pretrained("ckpt/tokenizer")
        >>> prompt = data_to_str(notes_to_sheet(midi_to_notes(mido.MidiFile("Touhou Broken_Moon.mid"))))
        >>> for token in generate_sheet(prompt, model, tokenizer, seed=1989, temperature=0.604):
        ...     prompt += token
        >>> generated_notes = sheet_to_notes(str_to_data(prompt))
        >>> mido.MidiFile(trakcs=[notes_to_track(generated_notes)]).save("generated.mid")
    """
    # 初始化随机数生成器并设置种子
    generator = torch.Generator(device=device).manual_seed(seed)

    # 编码提示文本并移除结束标记
    input_tensor = torch.tensor(tokenizer.encode(prompt)[:-1], device=device)

    # 自回归生成循环
    while True:
        # 模型前向计算
        with torch.no_grad():
            # 增加输入张量的批次维度，再进行推理
            logits = model(input_tensor.unsqueeze(0), torch.zeros((1, input_tensor.size())))[0, -1, :]

        # 屏蔽特殊标记(BOS/PAD/UNK)
        logits[tokenizer.bos_token_id] = -torch.inf
        logits[tokenizer.pad_token_id] = -torch.inf
        logits[tokenizer.unk_token_id] = -torch.inf

        # 应用温度参数并计算概率分布
        probs = F.softmax(logits / temperature, dim=-1)

        # 从概率分布中采样下一个标记
        next_token = torch.multinomial(probs, 1, generator=generator).item()

        # 遇到结束标记则停止生成
        if next_token == tokenizer.eos_token_id:
            break

        # 返回生成的标记
        yield tokenizer.convert_ids_to_tokens(next_token)

        # 将新标记添加到输入中用于下一次迭代
        input_tensor = torch.cat([input_tensor, torch.tensor([next_token], device=device)], dim=-1)


def generate_midi(
    prompt: list[tuple[int, int]],
    model: MidiNet,
    tokenizer: PreTrainedTokenizerFast,
    seed: int = 8964,
    temperature: float = 1.,
    device: torch.device = None
) -> Iterator[tuple[int, int]]:
    """
    根据输入的MIDI音符序列生成新的音乐内容。
    使用后台线程持续生成乐谱数据，通过流式缓冲区实时输出转换后的音符。

    工作流程:
    1. 将输入音符序列转换为乐谱事件格式
    2. 启动后台线程调用generate_sheet生成乐谱
    3. 将生成的乐谱事件实时转换为音符元组
    4. 通过生成器逐步输出每个音符

    Args:
        prompt: 起始音符序列，每个元素为(pitch, interval)元组
        model: 用于音乐生成的预训练模型
        tokenizer: 乐谱事件的分词器
        seed: 控制生成随机性的种子值
        temperature: 控制生成多样性的温度参数
        device: 模型运行的计算设备

    Returns:
        生成器，每次迭代返回一个生成的新音符(pitch, interval)

    Examples:
        >>> import mido
        >>> from utils import midi_to_notes, notes_to_sheet
        >>> from tokenizer import data_to_str, str_to_data
        >>> model = MidiNet(...)
        >>> model.load_state_dict(torch.load("ckpt/model.pth"))
        >>> tokenizer = PreTrainedTokenizerFast.from_pretrained("ckpt/tokenizer")
        >>> prompt = midi_to_notes(mido.MidiFile("Touhou Broken_Moon.mid"))
        >>> for pitch, interval in generate_midi(prompt, model, tokenizer):
        ...     print(f"生成音符: 音高{pitch}，与前一个音符的时间间隔{interval}")
    """
    # 创建线程安全的数据流缓冲区
    music_stream = BufferStream()

    def _generate_in_background():
        "后台生成线程的工作函数"
        # 将输入音符转换为乐谱字符串格式
        sheet_music = notes_to_sheet(prompt)
        prompt_text = data_to_str(sheet_music)

        # 持续生成乐谱标记并送入缓冲区
        for token in generate_sheet(prompt_text, model, tokenizer, seed, temperature, device):
            # 将标记转换为数据结构并发送到流
            music_stream.send(str_to_data(token))

        # 停止迭代器
        music_stream.stop()

    # 启动后台生成线程（守护线程确保主程序退出时自动结束）
    threading.Thread(_generate_in_background, daemon=True).start()

    # 从流中实时读取并转换为音符序列
    for note in sheet_to_notes(music_stream):
        # 将乐谱数据转换为音符并逐个产出
        yield note


def normalize_pitch_center(notes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    将音符序列的音高居中化处理，使平均音高移动到64附近。

    工作流程:
    1. 分离音高和时值序列
    2. 计算当前平均音高
    3. 调整所有音高使平均音高移动到目标值(64)
    4. 重新组合音高和时值

    Args:
        notes: 原始音符序列，每个元素为(pitch, interval)元组

    Returns:
        处理后的音符序列，音高整体平移

    Examples:
        >>> notes = [(60, 4), (64, 2)]  # 平均音高62
        >>> normalize_pitch_center(notes)
        [(62, 4), (66, 2)]  # 平均音高变为64
    """
    # 分离音高和时值（使用更安全的解包方式）
    try:
        pitches, intervals = zip(*notes)
    except ValueError:
        return []  # 处理空输入情况

    # 计算当前平均音高（使用浮点除法）
    current_avg = sum(pitches) / len(pitches)

    # 计算需要平移的半音数（目标音高64）
    pitch_shift = round(64 - current_avg)

    # 平移所有音高并重新组合（使用列表推导式更高效）
    return [
        (pitch + pitch_shift, interval)
        for pitch, interval in zip(pitches, intervals)
    ]


def main():
    # 加载模型的预训练检查点
    tokenizer, state = load_checkpoint(pathlib.Path("ckpt"), train=False)

    # 获取模型参数
    vocab_size, d_model = state["embedding.weight"].size()
    dim_feedforward = state["transformer.layers.0.linear1.weight"].size(0)
    num_heads = d_model // 64
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

    notes = normalize_pitch_center(generate_midi(LOVE_TRADING_MIDI, model, tokenizer, device=device))
    track = notes_to_track(notes)
    mido.MidiFile(tracks=[track]).save("example.mid")


if __name__ == "__main__":
    main()
