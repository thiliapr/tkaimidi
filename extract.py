"从 MIDI 文件夹中提取训练信息，以方便训练分词器和模型。"
# Copyright (C)  thiliapr 2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import json
import random
import multiprocessing
import argparse
import pathlib
import mido

# 根据是否在 Jupyter 环境下导入不同库
if "get_ipython" in globals():
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
    from utils import midi_to_notes, notes_to_sheet
    from tokenizer import data_to_str


def process_midi_to_json(
    rank: int,
    midi_files: list[pathlib.Path],
    root_dir: pathlib.Path,
    output_dir: pathlib.Path,
    max_sequence_length: int,
    min_sequence_length: int
):
    """
    将 MIDI 文件转换为训练用的 JSON 格式

    处理流程:
    1. 读取 MIDI 文件并解析音符
    2. 过滤无效或过短的音符序列
    3. 将音符转换为电子乐谱表示
    4. 根据长度要求截断或跳过序列
    5. 将结果保存为 JSON 文件

    Args:
        rank: 进程标识符，用于显示进度条
        midi_files: 要处理的 MIDI 文件列表
        root_dir: 输入文件的根目录
        output_dir: 输出 JSON 文件的目录
        max_sequence_length: 最大允许的序列长度
        min_sequence_length: 最小允许的音符数量
    """
    print(f"进程 {rank} 已启动。")
    for filepath in tqdm(midi_files, desc=f"进程 {rank}", position=rank):
        try:
            # 读取 MIDI 文件，clip=True 自动处理异常事件
            midi_file = mido.MidiFile(filepath, clip=True)
        except Exception:
            continue

        # 提取音符序列并跳过小于指定长度的 MIDI 文件
        notes = midi_to_notes(midi_file)
        if len(notes) < min_sequence_length:
            continue

        # 转化为电子乐谱形式
        sheet, positions = notes_to_sheet(notes, max_length=max_sequence_length)

        # 跳过小于指定长度的 MIDI 文件
        if len(positions) < min_sequence_length:
            continue

        # 构建输出路径，保持原始目录结构
        relative_path = filepath.relative_to(root_dir)
        output_path = output_dir / relative_path.parent / (filepath.stem + ".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为紧凑的 JSON 格式
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "num_notes": len(notes),
                "positions": positions,
                "data": data_to_str(sheet)
            }, f, separators=(",", ":"))


def main():
    """
    MIDI 文件批量转换的主程序

    工作流程:
    1. 解析命令行参数
    2. 收集所有 MIDI 文件
    3. 分配任务到多个进程
    4. 并行处理文件转换
    """
    parser = argparse.ArgumentParser(description="从 MIDI 文件夹中提取训练信息")
    parser.add_argument("input_dir", type=pathlib.Path, help="要提取的 MIDI 文件夹。")
    parser.add_argument("output_dir", type=pathlib.Path, help="MIDI 信息输出文件夹。")
    parser.add_argument("-m", "--min-sequence-length", default=128, type=int, help="最小序列长度，小于该长度的样本不会被转换（单位: 音符）")
    parser.add_argument("-e", "--max-sequence-length", default=2 ** 17, type=int, help="最大序列长度，大于该长度的样本将被截断（单位: 字符）")
    parser.add_argument("-j", "--jobs", type=int, help="并行工作进程数（默认: 使用所有CPU核心）")
    args = parser.parse_args()

    # 获取并行进程数
    n_jobs = args.jobs
    if not n_jobs:
        n_jobs = multiprocessing.cpu_count()
    print(f"使用 {n_jobs} 个进程并行处理")

    # 遍历输入目录中的所有 MIDI 文件
    midi_files = list(args.input_dir.glob("**/*.mid"))
    print(f"发现 {len(midi_files)} 个 MIDI 文件")

    # 随机打乱文件顺序以实现负载均衡
    random.shuffle(midi_files)

    # 分配任务批次
    batch_size = (len(midi_files) + n_jobs - 1) // n_jobs
    batches = [midi_files[i:i + batch_size] for i in range(0, len(midi_files), batch_size)]

    # 准备多进程参数
    task_args = [
        (rank, batch, args.input_dir, args.output_dir, args.max_sequence_length, args.min_sequence_length)
        for rank, batch in enumerate(batches)
    ]

    # 使用进程池并行处理
    with multiprocessing.Pool() as pool:
        pool.starmap(process_midi_to_json, task_args)


if __name__ == "__main__":
    main()
