# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

# 基本常识
TIME_PRECISION = 120  # 时间精度
PITCH_RANGE = 24

# 模型参数
DEFAULT_DIM_HEAD = 48
DEFAULT_NUM_HEADS = 8
DEFAULT_DIM_FEEDFORWARD = DEFAULT_DIM_HEAD * DEFAULT_NUM_HEADS * 8 // 3
DEFAULT_NUM_LAYERS = 12  # GPTBlock 堆叠层数

# 训练参数
DEFAULT_LEARNING_RATE = 1e-5  # 学习率
DEFAULT_WEIGHT_DECAY = 1e-2  # 权重衰减系数
DEFAULT_DROPOUT = 0  # Dropout 率
DEFAULT_ACCUMULATION_STEPS = 8  # 梯度累计步数
DEFAULT_PROBABILITY_MAPS_LENGTH = 64
DEFAULT_TRAIN_MAX_BATCH_TOKENS = 2048  # 训练时，每个批次的序列长度的和上限
DEFAULT_VAL_MAX_BATCH_TOKENS = 4096  # 训练时，每个批次的序列长度的和上限
DEFAULT_PITCH_PERTURB_PROB = 0.1  # 输入序列中每个音符被随机替换音高的最大概率因子