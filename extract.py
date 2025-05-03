"从 MIDI 文件夹中提取训练信息，以方便训练分词器和模型。"
# Copyright (C)  thiliapr 2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import argparse
import pathlib
import mido
import json

# 根据是否在 Jupyter 环境下导入不同库
if "get_ipython" in globals():
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
    from utils import midi_to_notes, notes_to_sheet
    from tokenizer import data_to_str


def main():
    parser = argparse.ArgumentParser(description="从 MIDI 文件夹中提取训练信息")
    parser.add_argument("input_dir", type=pathlib.Path, help="要提取的 MIDI 文件夹。")
    parser.add_argument("output_dir", type=pathlib.Path, help="MIDI 信息输出文件夹。")
    parser.add_argument("-m", "--min-sequence-length", default=128, type=int, help="最小序列长度，小于该长度的样本不会被转换（单位: 音符）")
    parser.add_argument("-e", "--max-sequence-length", default=2 ** 17, type=int, help="最大序列长度，大于该长度的样本将被截断（单位: 字符）")
    args = parser.parse_args()

    # 遍历输入目录中的所有 MIDI 文件
    for filepath in tqdm(list(args.input_dir.glob("**/*.mid"))):
        # 读取并转化 MIDI 文件
        try:
            midi_file = mido.MidiFile(filepath, clip=True)
        except Exception:
            # 跳过有错误的 MIDI 文件
            continue

        # 提取音符并跳过没有音符的 MIDI 文件
        notes = midi_to_notes(midi_file)
        if not notes:
            continue

        # 转化为电子乐谱形式
        sheet, positions = notes_to_sheet(notes)

        # 截断超长序列
        if len(sheet) > args.max_sequence_length:
            notes_end, sheet_end = max((i, position) for i, position in enumerate(positions) if position < args.max_sequence_length)
            notes = notes[:notes_end]
            sheet = sheet[:sheet_end]

        # 跳过过短序列
        if len(notes) < args.min_sequence_length:
            continue

        # 过滤出可以用于训练的偏移量
        train_notes = [i for i, (_, interval) in enumerate(notes) if i == 0 or interval != 0]

        # 构建输出路径并确保目录存在
        output_path = args.output_dir / filepath.relative_to(args.input_dir).parent / (filepath.name[:-3] + "json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 将提取的信息保存为 JSON 文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "num_notes": len(notes),
                "train_notes": train_notes,
                "positions": [position for i, position in enumerate(positions) if i in train_notes],
                "data": data_to_str(sheet)
            }, f)


if __name__ == "__main__":
    main()
