"音乐分词器及模型训练使用的常量集"

# 本文件是 tkaimidi 的一部分
# SPDX-FileCopyrightText: 2024-2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

# 基本常识
TIME_PRECISION = 120  # 时间精度
BASE_CHAR_CODE = ord("A")  # 基础字符编码，用于将整数0-16映射到字符
NATURAL_SCALE = {0, 2, 4, 5, 7, 9, 11}  # 自然大调音阶对应的半音位置（C大调）

# 电子乐谱
# 事件
KEY_UP = 12  # 全局音高向上调整，适用于该事件以后到下一个KEY_UP或KEY_DOWN的所有音符事件
KEY_DOWN = 13  # 全局音高向下调整，适用于该事件以后到下一个KEY_UP或KEY_DOWN的所有音符事件
OCTAVE_JUMP_UP = 14  # 八度向上跳跃，仅适用于该事件以后的下一个音符事件
OCTAVE_JUMP_DOWN = 15  # 八度向下跳跃，仅适用于该事件以后的下一个音符事件
TIME_INTERVAL = 16  # 时间停顿，进入下一个时间刻
# 视野大小，用于调整音高范围保证转换出来的音高大多在自然音阶内，过大容易忽视局部变化，过小容易转换出过多音调变化事件
LOOKAHEAD_COUNT = 64

# 模型参数
DEFAULT_DIM_HEAD = 64  # 注意力头的维度
DEFAULT_NUM_HEADS = 16  # 注意力头的数量
DEFAULT_DIM_FEEDFORWARD = DEFAULT_DIM_HEAD * DEFAULT_NUM_HEADS * 4  # 前馈层的维度。根据经验，dim_feedforward = dim_head * num_heads * 4
DEFAULT_NUM_LAYERS = 12  # Transformer 层数

# 训练参数
DEFAULT_LEARNING_RATE = 5e-5  # 学习率
DEFAULT_WEIGHT_DECAY = 1e-2  # 权重衰减（L2正则化）系数
DEFAULT_DROPOUT = 0.1  # Dropout 概率
DEFAULT_MIN_SEQUENCE_LENGTH = 64  # 最小序列长度。小于该长度的序列不会被训练。选择 64 作为下限是为了确保每个训练样本包含足够的上下文信息，有助于模型学习音乐结构；过短的序列可能导致模型难以捕捉长期依赖，影响生成质量。
