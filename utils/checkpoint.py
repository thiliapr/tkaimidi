# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib
from typing import Any, TypedDict
from collections.abc import Mapping
import orjson
import torch
from utils.model import MidiNetConfig


class MidiNetInfo(TypedDict):
    """
    模型的额外信息，也就是不能从状态字典中推断出来的信息

    Args:
        pitch_num_heads: 音高特征编码器注意力头的数量
        num_heads: 编-解码器注意力头的数量
        completed_epochs: 总共训练了多少个 Epoch
    """
    pitch_num_heads: int
    num_heads: int
    completed_epochs: int


def save_checkpoint(
    path: pathlib.Path,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    scaler_state: Mapping[str, Any],
    ckpt_info: MidiNetInfo
):
    """
    保存模型的检查点到指定路径，包括模型的权重以及训练的进度信息。

    Args:
        path: 保存检查点的目录路径
        model_state: 要保存的模型的状态字典
        optimizer_state: 要保存的优化器的状态字典
        scaler_state: 要保存的梯度缩放器的状态字典
        ckpt_info: 模型额外信息（不能从状态字典推断出的信息）
    """
    path.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在，如果不存在则创建
    torch.save(model_state, path / "model.pth")  # 保存模型权重
    torch.save(optimizer_state, path / "optimizer.pth")  # 保存优化器权重
    torch.save(scaler_state, path / "scaler.pth")  # 保存缩放器权重

    # 保存额外信息
    (path / "info.json").write_bytes(orjson.dumps(ckpt_info))


def load_checkpoint(path: pathlib.Path) -> tuple[Mapping[str, Any], MidiNetConfig, MidiNetInfo]:
    """
    从指定路径加载模型的检查点（用于推理）。

    Args:
        path: 加载检查点的目录路径

    Returns:
        模型的状态、用于创建模型的配置、额外信息

    Examples:
        >>> state_dict, model_config, ckpt_info = load_checkpoint(pathlib.Path("ckpt"))
        >>> model = MidiNet(model_config, device=torch.device("cuda"))
        >>> model.load_state_dict(state_dict)
    """
    # 加载模型权重
    model_state = torch.load(path / "model.pth", weights_only=True, map_location=torch.device("cpu"))

    # 加载模型额外
    ckpt_info = orjson.loads((path / "info.json").read_bytes())

    # 提取模型配置
    model_config = extract_config(model_state, ckpt_info["pitch_num_heads"], ckpt_info["num_heads"])
    return model_state, model_config, ckpt_info


def load_checkpoint_train(path: pathlib.Path) -> tuple[Mapping[str, Any], MidiNetConfig, MidiNetInfo, Mapping[str, Any], Mapping[str, Any]]:
    """
    从指定路径加载模型的检查点（用于恢复训练状态）。

    Args:
        path: 加载检查点的目录路径

    Returns:
        模型的状态、用于创建模型的配置、模型额外信息、优化器的状态、梯度缩放器状态

    Examples:
        >>> model_state, model_config, ckpt_info, optimizer_state, scaler_state = load_checkpoint_train(pathlib.Path("ckpt"))
        >>> model = MidiNet(model_config, deivce="cuda")
        >>> model.load_state_dict(model_state)
        >>> optimizer = optim.AdamW(model.parameters())
        >>> optimizer.load_state_dict(optimizer_state)
        >>> scaler = torch.amp.GradScaler("cuda")
        >>> scaler.load_state_dict(scaler_state)
    """
    # 加载检查点
    model_state, model_config, ckpt_info = load_checkpoint(path)

    # 加载优化器、缩放器权重
    optimizer_state = torch.load(path / "optimizer.pth", weights_only=True, map_location=torch.device("cpu"))
    scaler_state = torch.load(path / "scaler.pth", weights_only=True, map_location=torch.device("cpu"))

    # 返回训练所需信息
    return model_state, model_config, ckpt_info, optimizer_state, scaler_state


def extract_config(model_state: dict[str, Any], pitch_num_heads: int, num_heads: int) -> MidiNetConfig:
    """
    从模型状态字典中提取 MidiNet 模型的配置参数
    通过分析 state_dict 中各层的维度大小和结构，自动推断出模型的超参数配置

    Args:
        model_state: 保存模型参数的状态字典
        pitch_num_heads: 音高特征编码器的注意力头数量
        num_heads: 编-解码器的注意力头数量

    Returns:
        包含所有提取出的配置参数的 MidiNetConfig 对象

    Examples:
        >>> state_dict = torch.load("model.pth")
        >>> config = extract_config(state_dict, 1)  # 假设单头注意力
    """
    pitch_dim_head = model_state["note_embedding"].size(0) // pitch_num_heads
    pitch_dim_feedforward, _, pitch_conv1_kernel = model_state["pitch_feature_encoder.0.conv1.weight"].shape
    pitch_conv2_kernel = model_state["pitch_feature_encoder.0.conv2.weight"].size(2)
    dim_head = model_state["pitch_projection.weight"].size(0) // num_heads
    dim_feedforward = model_state["encoder.0.linear1.weight"].size(0)
    varaince_bins = model_state["pitch_mean_embedding.weight"].size(0)
    num_pitch_layers = len({key.split(".")[1] for key in model_state if key.startswith("pitch_feature_encoder.")})
    num_note_count_layers = len({key.split(".")[2] for key in model_state if key.startswith("note_count_predictor.layers.")})
    num_pitch_mean_layers = len({key.split(".")[2] for key in model_state if key.startswith("pitch_mean_predictor.layers.")})
    num_pitch_range_layers = len({key.split(".")[2] for key in model_state if key.startswith("pitch_range_predictor.layers.")})
    num_encoder_layers = len({key.split(".")[1] for key in model_state if key.startswith("encoder.")})
    num_decoder_layers = len({key.split(".")[1] for key in model_state if key.startswith("decoder.")})
    return MidiNetConfig(pitch_num_heads, pitch_dim_head, pitch_dim_feedforward, num_heads, dim_head, dim_feedforward, pitch_conv1_kernel, pitch_conv2_kernel, varaince_bins, num_pitch_layers, num_note_count_layers, num_pitch_mean_layers, num_pitch_range_layers, num_encoder_layers, num_decoder_layers)
