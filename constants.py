"音乐分词器及模型训练使用的常量集"

# Copyright (C)  thiliapr 2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

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
DEFAULT_DIM_FEEDFORWARD = 4096  # 前馈层的维度。根据经验，dim_feedforward = dim_head * num_heads * 4
DEFAULT_NUM_LAYERS = 6  # Transformer 层数

# 训练参数
DEFAULT_LEARNING_RATE = 5e-5  # 学习率
DEFAULT_WEIGHT_DECAY = 1e-2  # 权重衰减（L2正则化）系数
DEFAULT_DROPOUT = 0.1  # Dropout 概率
DEFAULT_MIN_SEQUENCE_LENGTH = 64  # 最小序列长度。小于该长度的序列不会被训练
