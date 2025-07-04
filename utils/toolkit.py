"这个模块提供了一些实用工具函数，用于并行处理和内存管理。"

# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Optional
from collections.abc import Callable


def parallel_map(func: Callable, iterable: list[tuple], num_workers: Optional[int] = None):
    """
    使用多进程并行执行函数。
    该函数将给定的函数应用于可迭代对象的每个元素，并使用指定数量的工作进程并行处理。

    Args:
        func: 要应用的函数，接受可迭代对象的元素作为参数
        iterable: 可迭代对象，包含要处理的数据
        num_workers: 工作进程数量，默认为CPU核心数

    Returns:
        包含函数应用结果的列表
    """
    from multiprocessing import Pool, cpu_count
    num_workers = num_workers or cpu_count()  # 获取CPU核心数或指定数量
    with Pool(processes=num_workers) as pool:
        return pool.starmap(func, iterable)


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
