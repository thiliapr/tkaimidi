"训练分词器、数据转换"

# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizers的并行处理，避免冲突

import random
import argparse
import pathlib
import orjson
from typing import Iterable, Iterator, Any, Union
import mido
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from utils.constants import BASE_CHAR_CODE
from utils.midi import midi_to_notes, notes_to_sheet
from utils.toolkit import empty_cache, parallel_map


def data_to_str(data: Iterable[int]):
    """
    将整数列表转换为字符串表示

    Args:
        data: 整数列表，表示字符在字母表中的索引

    Returns:
        转换后的字符串
    """
    return "".join(chr(BASE_CHAR_CODE + char) for char in data)


def str_to_data(string: str):
    """
    将字符串转换回整数列表

    Args:
        string: 转换后的字符串

    Returns:
        转换后的数据。
    """
    return [ord(char) - BASE_CHAR_CODE for char in string]


def train_sheet_tokenizer(model_data_samples: Iterable[str], vocab_size: int, min_frequency: int):
    """
    训练专门用于处理模型数据的 tokenizer

    Args:
        model_data_samples: 多个经过 sheet_to_model 和 data_to_str 转换后的模型数据样本
        vocab_size: 词汇表大小，控制分词器的容量
        min_frequency: 最小出现频率，低于此值的token将被忽略

    Returns:
        训练好的 tokenizer 实例
    """
    # 初始化BPE分词器
    tokenizer = Tokenizer(models.BPE())

    # 准备训练器配置
    trainer = trainers.BpeTrainer(
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=[chr(BASE_CHAR_CODE + data) for data in range(17)],
        show_progress=False
    )

    # 使用样本数据训练分词器
    tokenizer.train_from_iterator(model_data_samples, trainer=trainer)

    # 配置后处理模板，添加开始和结束标记
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    # 包装
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    return wrapped_tokenizer


def get_samples_midi(rank: int, midi_files: list[pathlib.Path], max_sequence_length: int) -> list[tuple[int, str]]:
    """
    从 MIDI 文件中提取音符数据，并转换为字符串表示

    Args:
        rank: 进程编号，用于进度条显示
        midi_files: MIDI 文件的路径列表
        max_sequence_length: 最大序列长度，超过此长度的序列将被截断
    
    Returns:
        包含音符数量和字符串表示的元组列表
    """
    # 用于存储结果的列表
    result = []

    # 遍历 MIDI 文件夹
    for filepath in tqdm(midi_files, desc="处理 MIDI 文件", disable=rank != 0, delay=0.1):
        # 转换处理流程: MIDI 文件 → 音符 → 乐谱表示 → 字符表示
        try:
            midi_file = mido.MidiFile(filepath, clip=True)
        except (ValueError, EOFError, OSError):
            # 跳过有错误的 MIDI 文件
            continue

        # 提取音符并跳过没有音符的 MIDI 文件
        notes = midi_to_notes(midi_file)
        if not notes:
            continue

        # 转化为电子乐谱形式
        sheet, _ = notes_to_sheet(notes, max_length=max_sequence_length)
        result.append((len(notes), data_to_str(sheet)))

    return result


def get_samples_json(rank: int, json_files: list[pathlib.Path], max_sequence_length: int) -> list[tuple[int, str]]:
    """
    从 JSON 文件中提取音符数据，并转换为字符串表示

    Args:
        rank: 进程编号，用于进度条显示
        json_files: JSON 文件的路径列表
        max_sequence_length: 最大序列长度，超过此长度的序列将被截断
    
    Returns:
        包含音符数量和字符串表示的元组列表
    """
    result = []

    for filepath in tqdm(json_files, desc="处理 JSON 文件", disable=rank != 0, delay=0.1):
        # 读取 JSON 文件
        with open(filepath, encoding="utf-8") as f:
            data = orjson.loads(f.read())

        # 截断超长序列
        if len(data["data"]) > max_sequence_length:
            # 找到最大不超过 max_sequence_length 的分割点
            # data["positions"] 是一个整数列表，表示每个音符在字符串中的位置
            # 我们需要找到最后一个位置小于 max_sequence_length 的音符
            notes_end, sheet_end = max(
                (i, position)
                for i, position in enumerate(data["positions"])
                if position < max_sequence_length
            )

            # 更新数据
            data["num_notes"] = notes_end
            data["data"] = data["data"][:sheet_end]

        result.append((data["num_notes"], data["data"]))

    return result


def get_samples(dirs: list[pathlib.Path], max_sequence_length: int) -> list[tuple[int, str]]:
    """
    从指定目录中提取 MIDI 和 JSON 文件的音符数据，并转换为字符串表示
    该函数会并行处理目录中的 MIDI 和 JSON 文件，提取音符数据，并将其转换为字符串表示。

    Args:
        dirs: 包含 MIDI 和 JSON 文件的目录列表
        max_sequence_length: 最大序列长度，超过此长度的序列将被截断

    Returns:
        包含音符数量和字符串表示的元组列表
    """
    # 收集所有 MIDI 文件路径
    midi_files = [
        f
        for midi_dir in dirs
        for f in midi_dir.rglob("*.*")
        if f.suffix.lower() in {".mid", ".midi"}
    ]

    # 收集所有 JSON 文件路径
    json_files = [
        f
        for json_dir in dirs
        for f in json_dir.rglob("*.*")
        if f.suffix.lower() == ".json"
    ]

    num_workers = os.cpu_count()  # 并行处理的进程数

    # 并行处理 MIDI 文件，提取样本
    midi_result = parallel_map(
        get_samples_midi,
        [(i, midi_files[i::num_workers], max_sequence_length) for i in range(num_workers)],
        num_workers=num_workers
    )
    # 并行处理 JSON 文件，提取样本
    json_result = parallel_map(
        get_samples_json,
        [(i, json_files[i::num_workers], max_sequence_length) for i in range(num_workers)],
        num_workers=num_workers
    )

    # 合并所有结果为一个列表
    result = [sample for sublist in midi_result + json_result for sample in sublist]
    return result


def validate(samples: Iterator[tuple[int, str]], tokenizer: PreTrainedTokenizerFast) -> dict[str, Union[int, float]]:
    """
    评估分词器的效果

    Args:
        samples: 样本列表，每个样本包含音符数量和字符串表示
        tokenizer: 要评估的分词器

    Returns:
        字典，包含:
        - 总音符数量
        - 平均序列长度占音符总数的百分比
        - 使用的词汇占总词汇表的比例
    """
    # 初始化统计变量
    total_notes = 0
    total_seq_length = 0
    words_used_set = set()

    # 遍历样本进行评估
    for num_notes, data in tqdm(samples, desc="评估效果"):
        total_notes += num_notes
        encoded = tokenizer.encode(data)
        total_seq_length += len(encoded)
        words_used_set.update(encoded)

    # 统计
    avg_seq_ratio = total_seq_length / total_notes if total_notes > 0 else 0
    vocab_usage = (len(words_used_set) - 2) / (len(tokenizer) - 2)  # 减去2个特殊token

    return {"total_notes": total_notes, "avg_seq_ratio": avg_seq_ratio, "vocab_usage": vocab_usage}


def print_validation_results(metrics: dict[str, Any]):
    """
    以易读格式打印验证结果

    参数:
        metrics: validate函数返回的评估指标字典
    """
    print(f"- 总音符数: {metrics['total_notes']}")
    print(f"- 平均序列长度/音符数: {metrics['avg_seq_ratio']:.2%}")
    print(f"- 词汇表使用率: {metrics['vocab_usage']:.2%}")


def main():
    "主函数，处理命令行参数并执行分词器训练流程"
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="音乐数据分词器训练脚本")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="分词器保存路径，将创建tokenizer子目录")
    parser.add_argument("-t", "--train-samples", type=pathlib.Path, action="append", required=True, help="训练集目录路径，包含MIDI样本文件")
    parser.add_argument("-v", "--valid-samples", type=pathlib.Path, action="append", help="验证集目录路径，包含MIDI样本文件")
    parser.add_argument("-c", "--train-samples-count", type=int, default=2 ** 15, help="训练集样本数量，默认为 %(default)s")
    parser.add_argument("-n", "--valid-samples-count", type=int, default=2 ** 15, help="验证集样本数量，默认为 %(default)s")
    parser.add_argument("-u", "--seed", type=int, default=8964, help="随机种子，默认为 %(default)s")
    parser.add_argument("-s", "--vocab-size", type=int, default=10000, help="分词器词汇表大小，默认为 %(default)s")
    parser.add_argument("-f", "--min-frequency", type=int, default=24, help="token最小出现频率阈值，默认为 %(default)s")
    parser.add_argument("-m", "--max-sequence-length", type=int, default=2 ** 17, help="最长允许多长的序列参与训练和测试（单位: 字符），默认为 %(default)s")
    parser.add_argument("-y", "--force", action="store_true", help="即使检查点已经存在分词器也要训练新的分词器")

    # 解析参数
    args = parser.parse_args()
    tokenizer_path = args.ckpt_path / "tokenizer"

    # 如果检查点已有分词器，且未指定重新训练分词器，则跳过
    if tokenizer_path.exists() and not args.force:
        print("已存在分词器，跳过训练。如果想重新训练分词器，请在参数指定`-y/--force`")
        return

    # 检查并创建检查点目录
    args.ckpt_path.mkdir(parents=True, exist_ok=True)

    # 处理所有MIDI样本文件
    train_samples: list[tuple[int, str]] = get_samples(args.train_samples, max_sequence_length=args.max_sequence_length)
    
    # 如果训练样本数量超过指定数量，则随机抽样
    if len(train_samples) > args.train_samples_count:
        print(f"训练样本数量超过 {args.train_samples_count}，随机抽样...")
        train_samples = random.Random(args.seed).sample(train_samples, args.train_samples_count)

    # 清除缓存
    empty_cache()

    # 训练分词器
    print("开始训练分词器...")
    tokenizer = train_sheet_tokenizer(
        (sample for _, sample in train_samples),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )

    # 保存分词器
    tokenizer.save_pretrained(tokenizer_path)
    print(f"分词器已保存到 {tokenizer_path}")

    # 输出基本信息
    print("\n训练结果:")
    print(f"- 词汇表大小: {tokenizer.vocab_size}")
    print(f"- 训练样本数: {len(train_samples)}")

    # 清除缓存
    empty_cache()

    # 评估训练集效果
    print("\n训练集评估:")
    train_metrics = validate(train_samples, tokenizer)
    print_validation_results(train_metrics)

    # 清除缓存
    empty_cache()

    # 如果有验证集，评估验证集效果
    if args.valid_samples:
        print("\n验证集评估:")
        valid_samples = get_samples(args.valid_samples, max_sequence_length=args.max_sequence_length)
        if len(valid_samples) > args.valid_samples_count:
            print(f"验证集样本数量超过 {args.valid_samples_count}，随机抽样...")
            valid_samples = random.Random(args.seed).sample(valid_samples, args.valid_samples_count)

        valid_metrics = validate(valid_samples, tokenizer)
        print_validation_results(valid_metrics)


if __name__ == "__main__":
    main()
