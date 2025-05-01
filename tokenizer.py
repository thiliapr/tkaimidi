"训练分词器、数据转换"

# Copyright (C)  thiliapr 2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import argparse
import pathlib
import json
import mido
from tokenizers import Tokenizer, models, trainers
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from typing import Iterator, Any

# 解除限制
import os
from multiprocessing import cpu_count
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count())

# 根据是否在 Jupyter 环境下导入不同库
if "get_ipython" in globals():
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from constants import BASE_CHAR_CODE
    from utils import midi_to_notes, notes_to_sheet, empty_cache
    from tqdm import tqdm


def data_to_str(data: Iterator[int]):
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


def train_sheet_tokenizer(model_data_samples: Iterator[str], vocab_size: int, min_frequency: int):
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

    # 包装为 transformers 兼容的 tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    return wrapped_tokenizer


def get_samples(midi_dirs: "list[pathlib.Path]" = []) -> Iterator[tuple[int, str]]:
    """
    从 MIDI 文件中提取样本，转换为模型输入所需的音符序列。

    该函数会遍历给定的 MIDI 文件夹列表，提取 MIDI 文件中的音符信息，并将其转换为
    字符串格式以供模型使用。同时，它也可以从已转换的 JSON 文件中加载样本。

    Args:
        midi_files: MIDI 文件夹列表，包含原始 MIDI 文件和已转换为 JSON 格式文件。

    Yields:
        返回一个元组，包含音符数量和对应的字符表示。
    """
    # 遍历 MIDI 文件夹，提取音符样本
    for filepath in tqdm([filepath for midi_dir in midi_dirs for filepath in midi_dir.glob("**/*.mid")], desc="加载原始 MIDI 样本", delay=0.1):
        # 转换处理流程: MIDI 文件 → 音符 → 乐谱表示 → 字符表示
        try:
            midi_data = mido.MidiFile(filepath, clip=True)
        except Exception:
            # 跳过有错误的 MIDI 文件
            continue
        notes = midi_to_notes(midi_data)
        sheet, _ = notes_to_sheet(notes)
        data = data_to_str(sheet)
        yield len(notes), data

    for filepath in tqdm([filepath for midi_dir in midi_dirs for filepath in midi_dir.glob("**/*.json")], desc="加载被转换的 MIDI 样本", delay=0.1):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        yield data["num_notes"], data["data"]


def validate(samples: list[tuple[int, str]], tokenizer: PreTrainedTokenizerFast) -> tuple[int, float, float]:
    """
    评估分词器的效果

    Args:
        samples: 样本列表，每个样本包含音符数量和字符串表示
        tokenizer: 要评估的分词器

    Returns:
        元组，包含:
        - 总音符数量
        - 平均序列长度占音符总数的百分比
        - 使用的词汇占总词汇表的比例
    """
    total_seq_length = 0
    words_used_set = set()

    for _, data in samples:
        encoded = tokenizer.encode(data)
        total_seq_length += len(encoded)
        words_used_set.update(encoded)

    total_notes = sum(num_notes for num_notes, _ in samples)
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
    parser.add_argument("-t", "--train-samples", type=pathlib.Path, action="append", required=True, help="训练集目录路径，包含MIDI样本文件（以`.mid`或`.json`结尾）")
    parser.add_argument("-v", "--valid-samples", type=pathlib.Path, help="验证集目录路径，包含MIDI样本文件（以`.mid`或`.json`结尾）")
    parser.add_argument("-p", "--ckpt-path", type=pathlib.Path, default=pathlib.Path("ckpt"), help="分词器保存路径，将创建tokenizer子目录")
    parser.add_argument("-s", "--vocab-size", type=int, default=2000, help="分词器词汇表大小")
    parser.add_argument("-f", "--min-frequency", type=int, default=12, help="token最小出现频率阈值")
    parser.add_argument("-y", "--force", action="store_true", help="即使检查点已经存在分词器也要训练")

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
    train_samples = list(get_samples(args.train_samples))

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
    print(f"\n训练结果:")
    print(f"- 词汇表大小: {tokenizer.vocab_size}")
    print(f"- 训练样本数: {len(train_samples)}")

    # 评估训练集效果
    print("\n训练集评估:")
    train_metrics = validate(train_samples, tokenizer)
    print_validation_results(train_metrics)

    # 如果有验证集，评估验证集效果
    if args.valid_samples:
        print("\n加载验证样本...")
        valid_samples = list(get_samples(args.valid_samples))

        print("\n验证集评估:")
        valid_metrics = validate(valid_samples, tokenizer)
        print_validation_results(valid_metrics)


if __name__ == "__main__":
    main()
