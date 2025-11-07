# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import numpy as np
from typing import Union


def empty_cache():
    """
    清理 CUDA 显存缓存并执行 Python 垃圾回收。

    本函数会先触发 Python 的垃圾回收机制，释放未被引用的内存。
    如果检测到有可用的 CUDA 设备，则进一步清理 CUDA 显存缓存，释放未被 PyTorch 占用但已缓存的 GPU 显存。

    Examples:
        >>> empty_cache()
    """
    import torch
    import gc

    # 执行 Python 垃圾回收
    gc.collect()

    # 检查是否有可用的 CUDA 设备
    if torch.cuda.is_available():
        # 仅在 CUDA 设备上调用 empty_cache()
        torch.cuda.empty_cache()


def create_padding_mask(sequences: list[torch.Tensor]) -> torch.BoolTensor:
    """
    创建用于序列填充的布尔掩码张量

    根据输入序列的长度信息生成一个二维布尔掩码，其中 True 值表示对应位置为填充位置。
    掩码的维度为 [批次大小, 最大序列长度]，便于后续的填充操作和注意力机制计算。

    Args:
        sequences: 包含多个序列张量的列表

    Returns:
        二维布尔掩码张量，标识填充位置

    Examples:
        >>> create_padding_mask([torch.rand(3, 5), torch.rand(2, 5)])
        tensor([[False, False, False],
                [False, False,  True]])
    """
    # 获取各序列长度
    sequence_lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=int)

    # 计算最大序列长度
    max_length = sequence_lengths.max().item()

    # 生成位置索引矩阵并与各序列长度比较，创建掩码
    position_indices = torch.arange(max_length, dtype=torch.long)

    # 通过广播机制比较位置索引与序列长度，生成填充掩码（True表示填充位置）
    return position_indices.unsqueeze(0) >= sequence_lengths.unsqueeze(1)
