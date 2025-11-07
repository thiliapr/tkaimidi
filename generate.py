# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
from typing import Optional
import mido
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.nn import functional as F
from utils.checkpoint import load_checkpoint
from utils.model import MidiNet
from utils.midi import midi_to_notes, notes_to_sequence, sequence_to_notes, notes_to_track


@torch.inference_mode()
def generate(model: MidiNet, prompt: torch.Tensor, num_frames: int, show_progress: bool = True):
    """
    使用 MidiNet 模型逐步生成音乐帧序列。

    该函数通过自回归方式生成指定数量的音乐帧。每步生成一帧，并将该帧作为下一时间步的输入。
    生成过程使用 KV Cache 优化，避免重复计算。

    Args:
        model: 用于音乐生成的神经网络模型
        prompt: 初始音乐提示序列，作为生成的起点
        num_frames: 需要生成的帧数
        show_progress: 是否显示进度条

    Returns:
        包含完整生成序列和所有预测概率的元组

    Examples:
        >>> initial_prompt = torch.randn(1, 10, 88)
        >>> output, probability_maps = generate(model, prompt, 100)
    """
    # 初始化 KV Cache 和预测概率图
    kv_cache = None
    probability_maps = torch.empty((prompt.size(0), 0, model.embedding.weight.size(0)), device=prompt.device)

    # 逐步生成 num_frames 帧
    for _ in tqdm(range(num_frames), disable=not show_progress):
        # 首次使用完整提示，后续仅使用最后一帧
        model_input = prompt if kv_cache is None else prompt[:, -1:]
        logits, kv_cache = model(
            model_input,
            kv_cache=kv_cache
        )

        # 从候选采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs[:, -1], 1)

        # 将预测添加到序列中
        probability_maps = torch.cat([probability_maps, probs], dim=1)
        prompt = torch.cat([prompt, next_token], dim=1)

    return prompt, probability_maps


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 可选的命令行参数列表，默认为 None，表示使用 sys.argv

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description="以指定 MIDI 为前面部分并生成音乐和保存。")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点的路径")
    parser.add_argument("num_frames", type=int, help="需要生成的帧数")
    parser.add_argument("output_path", type=pathlib.Path, help="MIDI 文件保存路径。生成的 MIDI 文件将会保存到这里。")
    parser.add_argument("-m", "--midi-path", type=pathlib.Path, help="指定的 MIDI 文件，将作为生成的音乐的前面部分。如果未指定，将从头开始生成。")
    parser.add_argument("-s", "--show-probability-maps", action="store_true", help="是否显示生成过程中音高均值和音高范围的钢琴卷帘图表")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型的预训练检查点
    state_dict, model_config, _ = load_checkpoint(args.ckpt_path)

    # 初始化模型并加载状态
    model = MidiNet(model_config)
    model.load_state_dict(state_dict)

    # 转移模型到设备并设置为评估模式
    model = model.to(device).eval()

    # 加载音乐生成的提示部分
    prompt_notes = [(0, 0)]  # 默认空提示音符列表
    if args.midi_path:
        try:
            prompt_notes = midi_to_notes(mido.MidiFile(args.midi_path))
        except Exception as e:
            print(f"加载指定的 MIDI 文件时出错: {e}\n将使用空提示音符生成")

    # 转换为钢琴卷轴、添加批次维度并开始生成
    prompt_sequence = torch.tensor(notes_to_sequence(prompt_notes), device=device)
    sequence, probability_maps = generate(
        model,
        prompt_sequence.unsqueeze(0),
        args.num_frames
    )

    # 如果需要，绘制预测图表
    if args.show_probability_maps:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        ax1.set_title("Probability Maps")
        ax2.set_title("Difference (Target - Predicted)")
        plt.colorbar(ax1.imshow(
            probability_maps.squeeze(0).T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, probability_maps.size(1), -1.5 - (probability_maps.size(-1) - 2) // 2, (probability_maps.size(-1) - 2) // 2 + 0.5],
        ), ax=ax1)
        plt.colorbar(ax2.imshow(
            (F.one_hot(prompt_sequence[1:], num_classes=probability_maps.size(2)) - probability_maps[0, :prompt_sequence.size(0) - 1]).T,
            aspect="auto",
            origin="lower",
            cmap=LinearSegmentedColormap.from_list("b_white_r", ["blue", "black", "red"]),
            vmin=-1, vmax=1,
            extent=[0, prompt_sequence.size(0) - 1, -1.5 - (probability_maps.size(-1) - 2) // 2, (probability_maps.size(-1) - 2) // 2 + 0.5],
        ), ax=ax2)
        plt.show()

    # 转换为 MIDI 轨道并保存为文件
    track = notes_to_track(sequence_to_notes(sequence.squeeze(0).tolist()))
    mido.MidiFile(tracks=[track]).save(args.output_path)


if __name__ == "__main__":
    main(parse_args())
