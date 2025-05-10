# 定义模型
## `PositionalEncoding`
为序列中每个位置生成正弦-余弦位置编码。

### 参数表
| 参数名 | 类型 | 说明 | 默认值 |
| - | - | - | - |
| `model_dim` | `int` | 模型的隐藏维度，需为偶数 | — |

---

## `ScaleNorm`
执行基于 L2 范数的可学习缩放归一化。

### 参数表
| 参数名 | 类型 | 说明 | 默认值 |
| - | - | - | - |
| `dim` | `int` | 输入的维度，用于初始化缩放因子 | — |
| `eps` | `float` | 防止除零的小常数 | 1e-5 |
| `device` | `torch.device` | 放置参数的位置| None |

---

## `MidiNetLayer`
结合 FlashAttention、自定义归一化和前馈网络的 Transformer 层。

### 参数表
| 参数名 | 类型 | 说明 | 默认值 |
| - | - | - | - |
| `num_heads` | `int` | 注意力头的数量 | — |
| `head_dim` | `int` | 每个注意力头的维度 | — |
| `feedforward_dim` | `int` | 前馈网络隐藏层的维度 | — |
| `dropout` | `float` | Dropout 概率  | 0.0 |
| `device` | `torch.device` | 模型所在的计算设备 | None |

---

## `MidiNet`
多层堆叠的 Transformer 架构模型，用于 MIDI token 生成。

### 参数表
| 参数名 | 类型 | 说明 | 默认值 |
| - | - | - | - |
| `vocab_size` | `int` | MIDI 事件的词汇表大小 | — |
| `num_heads` | `int` | 注意力头的数量 | — |
| `head_dim` | `int` | 每个注意力头的维度 | — |
| `feedforward_dim` | `int` | 前馈网络隐藏层维度 | — |
| `num_layers` | `int` | Transformer 层数  | — |
| `dropout` | `float` | Dropout 概率 | 0.1 |
| `device` | `torch.device` | 模型计算设备 | None |
