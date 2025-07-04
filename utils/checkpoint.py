"这个模块实现了 MidiNet 模型的检查点保存和加载功能，以及模型配置的提取。"

# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib
from typing import Any
import torch
import orjson
from transformers import PreTrainedTokenizerFast
from utils.model import MidiNetConfig


def save_checkpoint(model_state_dict: dict[str, Any], optimizer_state_dict: dict[str, Any], metrics: dict[str, list], path: pathlib.Path):
    """
    保存模型的检查点到指定路径，包括模型的权重以及训练的进度信息。

    Args:
        model_state_dict: 要保存的模型的状态字典
        optimizer_state_dict: 要保存的优化器的状态字典
        metrics: 指标
        path: 保存检查点的目录路径
    """
    path.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在，如果不存在则创建
    torch.save(model_state_dict, path / "model.pth")  # 保存模型权重
    torch.save(optimizer_state_dict, path / "optimizer.pth")  # 保存优化器权重

    # 保存训练信息
    with open(path / "metrics.json", "wb") as f:
        f.write(orjson.dumps(metrics))  # 写入JSON文件


def load_checkpoint(path: pathlib.Path) -> tuple[PreTrainedTokenizerFast, dict[str, Any]]:
    """
    从指定路径加载模型的检查点（用于推理）。

    Args:
        path: 加载检查点的目录路径

    Returns:
        分词器、模型的状态
    
    Examples:
        >>> tokenizer, sd = load_checkpoint(pathlib.Path("ckpt"))
        >>> config = extract_midi_net_config(sd)
        >>> model = MidiNet(config, device=torch.device("cuda"))
        >>> model.load_state_dict(sd)
    """
    # 加载分词器
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path / "tokenizer")

    # 检查并加载模型权重
    model_state = {}
    if (model_path := path / "model.pth").exists():
        # 加载模型的状态字典并更新
        model_state = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))  # 从检查点加载权重

    return tokenizer, model_state


def load_checkpoint_train(path: pathlib.Path) -> tuple[PreTrainedTokenizerFast, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    从指定路径加载模型的检查点（用于恢复训练状态）。

    Args:
        path: 加载检查点的目录路径
        train: 是否加载训练所需数据（优化器状态等）

    Returns:
        分词器、模型和优化器的状态，指标
    
    Examples:
        >>> tokenizer, msd, osd, metrics = load_checkpoint_train(pathlib.Path("ckpt"))
        >>> config = extract_midi_net_config(msd)
        >>> model = MidiNet(config, deivce=torch.device("cuda"))
        >>> model.load_state_dict(msd)
        >>> optimizer = optim.AdamW(model.parameters())
        >>> 
    """
    # 加载分词器和模型状态
    tokenizer, model_state = load_checkpoint(path)
    
    # 检查并加载优化器权重
    optimizer_state = {}
    if (optimizer_path := path / "optimizer.pth").exists():
        optimizer_state = torch.load(optimizer_path, weights_only=True, map_location=torch.device("cpu"))  # 从检查点加载权重

    # 尝试加载指标文件
    metrics_path = path / "metrics.json"
    metrics = {"val_loss": [], "train_loss": []}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics |= orjson.loads(f.read())  # 读取指标

    return tokenizer, model_state, optimizer_state, metrics


def extract_config(state_dict: dict[str, Any]) -> MidiNetConfig:
    """
    从 MidiNet 的 state_dict 中提取模型的结构配置信息。

    该函数通过解析模型的权重形状，自动推断出 MidiNet 的配置参数，
    包括词表大小、注意力头数量、每个头的维度、前馈层维度，以及网络层数。

    工作流程:
    1. 读取嵌入层的权重形状，确定词表大小和模型总维度 dim_model。
    2. 读取注意力层中合并的 qkv 权重，计算出 dim_head（每个注意力头的维度）。
    3. 反推出 num_heads（注意力头数量）。
    4. 读取前馈网络第一层的输出维度，得到 dim_feedforward。
    5. 统计 transformer 层（MidiNetLayer）的层数 num_layers。

    Args:
        state_dict: 模型的 state_dict。

    Returns:
        包含 MidiNet 结构参数的 MidiNetConfig 实例。

    Examples:
        >>> tokenizer, sd = load_checkpoint(pathlib.Path("ckpt"))
        >>> config = extract_config(sd)
        >>> model = MidiNet(config, deivce=torch.device("cuda"))
        >>> model.load_state_dict(sd)
    """
    vocab_size, dim_model = state_dict["embedding.weight"].size()
    dim_feedforward = state_dict["layers.0.feedforward.0.weight"].size(0)
    dim_head = (state_dict["layers.0.attention.qkv_proj.weight"].size(0) - dim_model) // 2
    num_heads = dim_model // dim_head
    num_layers = len(set(key.split(".")[1] for key in state_dict.keys() if key.startswith("layers.")))
    return MidiNetConfig(vocab_size, num_heads, dim_head, dim_feedforward, num_layers)
