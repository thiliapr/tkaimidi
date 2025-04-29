"训练检查点的加载、保存。"

# Copyright (C)  thiliapr 2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

import json
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import PreTrainedTokenizerFast

# 在非 Jupyter 环境下导入模型库
if "get_ipython" not in globals():
    from model import MidiNet


def save_checkpoint(model: MidiNet, optimizer: optim.AdamW, metrics: dict[str, list], path: pathlib.Path):
    """
    保存模型的检查点到指定路径，包括模型的权重以及训练的进度信息。

    Args:
        model: 要保存的模型实例
        optimizer: 要保存的优化器实例
        metrics: 指标
        path: 保存检查点的目录路径
    """
    path.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在，如果不存在则创建

    model = model.cpu()  # 将模型移到CPU进行保存
    # 处理DataParallel情况
    if isinstance(model, nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), path / "model.pth")  # 保存模型权重
    torch.save(optimizer.state_dict(), path / "optimizer.pth")  # 保存优化器权重

    # 保存训练信息
    with open(path / "metrics.json", "w") as f:
        json.dump(metrics, f)  # 写入JSON文件


def load_checkpoint(path: pathlib.Path, train: bool = False):
    """
    从指定路径加载模型的检查点，并恢复训练状态。

    Args:
        path: 加载检查点的目录路径
        train: 是否加载训练所需数据（优化器状态等）

    Returns:
        train关闭时: 分词器、模型的状态
        train启用时: 分词器、模型和优化器的状态，指标
    """
    # 加载分词器
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path / "tokenizer")

    # 检查并加载模型权重
    model_state = {}
    if (model_path := path / "model.pth").exists():
        # 加载模型的状态字典并更新
        model_state = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))  # 从检查点加载权重

    if not train:
        return tokenizer, model_state

    # 检查并加载优化器权重
    optimizer_state = {}
    if (optimizer_path := path / "optimizer.pth").exists():
        optimizer_state = torch.load(optimizer_path, weights_only=True, map_location=torch.device("cpu"))  # 从检查点加载权重

    # 尝试加载指标文件
    metrics_path = path / "metrics.json"
    metrics = {"val_ppl": [], "train_ppl": [], "val_loss": [], "train_loss": []}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics |= json.load(f)  # 读取指标

    return tokenizer, model_state, optimizer_state, metrics
