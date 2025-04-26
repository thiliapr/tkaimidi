"音乐分词器及模型训练使用的常量集"

# Copyright (C)  thiliapr 2025
# Email: thiliapr@tutanota.com
# 本文件是 tkaimidi 的一部分。
# tkaimidi 是自由软件：你可以再分发之和/或依照由自由软件基金会发布的 GNU Affero 通用公共许可证修改之，无论是版本 3 许可证，还是（按你的决定）任何以后版都可以。
# 发布 tkaimidi 是希望它能有用，但是并无保障；甚至连可销售和符合某个特定的目的都不保证。请参看 GNU Affero 通用公共许可证，了解详情。
# 你应该随程序获得一份 GNU Affero 通用公共许可证的复本。如果没有，请看 <https://www.gnu.org/licenses/>。

TIME_PRECISION = 120  # 时间精度
BASE_CHAR_CODE = ord("A")  # 基础字符编码，用于将整数0-16映射到字符
NATURAL_SCALE = {0, 2, 4, 5, 7, 9, 11}  # 自然大调音阶对应的半音位置（C大调）
