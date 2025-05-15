# 音乐生成工具使用文档

## 简介

该工具基于预训练的音乐生成模型，支持从指定的 MIDI 文件或者内置的示例 MIDI 音乐生成新的音乐。您可以通过命令行指定模型检查点、生成参数等，并将生成的音乐保存为 MIDI 文件。

## 使用示例 

```bash
python generate.py ./ckpt -m ./input.mid -t 0.8 -s 8964 -l 200
```

该命令会根据 `./ckpt` 检查点，使用 `./input.mid` 作为提示音乐，生成温度为 0.8，随机种子为 8964，最多生成 200 个音符的 MIDI 文件。

## 命令行参数

| 参数                               | 描述                                    | 默认值          |
| -------------------------------- | ------------------------------------- | ------------ |
| `ckpt_path`                      | 模型检查点的路径                              | 无            |
| `output_path`                    | MIDI 文件保存路径。生成的 MIDI 文件将会保存到这里。          | 无            |
| `-m, --midi-path`                | 用作提示的 MIDI 文件路径，如果未指定，则使用内置示例 MIDI 文件 | 内置示例 MIDI 文件 |
| `-t, --temperature`              | 生成音乐的采样温度参数，控制生成结果的多样性                | 1.0          |
| `-s, --seed`                     | 随机种子，指定随机种子用于生成控制                     | 随机生成         |
| `-p, --max-pitch-span-semitones` | 最大音高跨度，当音高变化超过此值时，自动调整音调生成概率          | 64           |
| `-l, --max-length`               | 最大生成音符数量，达到此数量后停止生成                   | 无限制            |
| `-k, --top-k`    				   | 仅对概率前`top_k`的token采样，减小随机性      | 无 |
| `-n, --num-heads`                | 模型的注意力头数量，控制模型的多头注意力层数                | 默认模型参数       |

## 函数说明

### `notes_to_track(notes: list[int]) -> mido.MidiTrack`

将音符和时间信息转换为 MIDI 轨道。

#### 参数

| 参数      | 类型          | 描述        |
| ------- | ----------- | --------- |
| `notes` | `list[int]` | 音符间隔格式的列表 |

#### 返回值

返回一个包含音符事件及结束标记的 MIDI 轨道。

---

### `generate_sheet(prompt: str, model: MidiNet, tokenizer: PreTrainedTokenizerFast, seed: int, temperature: float, device: torch.device) -> Generator[str, Optional[list[tuple[str, float]]], None`

使用自回归方式生成音乐乐谱事件序列的生成器函数。

#### 参数

| 参数            | 类型                        | 描述                       |
| ------------- | ------------------------- | ------------------------ |
| `prompt`      | `str`                     | 用于初始化生成的乐谱事件序列文本         |
| `model`       | `MidiNet`                 | 预训练的音乐生成模型实例             |
| `tokenizer`   | `PreTrainedTokenizerFast` | 用于乐谱事件与 token 互相转换的分词器实例 |
| `seed`        | `int`                     | 随机种子                     |
| `temperature` | `float`                   | 温度参数，控制生成多样性             |
| `top_k`		| `Optional[int]`           | 仅对概率前`top_k`个token采样，减小随机性 |
| `device`      | `torch.device`            | 计算设备                     |

#### 返回值

返回一个生成器，每次迭代生成一个乐谱事件 token 的字符串表示。

---

### `generate_midi(prompt: list[tuple[int, int]], model: MidiNet, tokenizer: PreTrainedTokenizerFast, seed: Optional[int] = None, temperature: float = 1., max_pitch_span_semitones: int = 64, max_length: Optional[int] = None, device: torch.device = None) -> Iterator[tuple[int, int]]`

实时流式生成 MIDI 音符序列。

#### 参数

| 参数                         | 类型                        | 描述                         |
| -------------------------- | ------------------------- | -------------------------- |
| `prompt`                   | `list[tuple[int, int]]`   | 初始音符序列，每个元素为 (音高, 间隔时间) 元组 |
| `model`                    | `MidiNet`                 | 预训练的音乐生成模型实例               |
| `tokenizer`                | `PreTrainedTokenizerFast` | 用于乐谱事件与文本互相转换的分词器          |
| `seed`                     | `Optional[int]`           | 随机种子，不指定表示随机生成             |
| `temperature`              | `float`                   | 控制生成多样性的温度参数               |
| `top_k`		             | `Optional[int]`           | 仅对概率前`top_k`个token采样，减小随机性 |
| `max_pitch_span_semitones` | `int`                     | 音高跨度超过该值时将进行音高调整           |
| `max_length`               | `Optional[int]`           | 限制最多生成的音符数量                |
| `device`                   | `torch.device`            | 模型运行的设备 (cpu/cuda)         |

#### 返回值

返回一个迭代器，生成每个音符的音高和时间间隔。

---

### `center_pitches(pitches: list[int]) -> list[tuple[int, int]]`

将音符序列的音高居中化处理，使平均音高移动到 64 附近。

#### 参数

| 参数        | 类型          | 描述   |
| --------- | ----------- | ---- |
| `pitches` | `list[int]` | 音高序列 |

#### 返回值

返回调整后的音高序列，音高整体平移，使得平均音高接近 64。

---

### `clamp_midi_pitch(pitches: list[int])`

将音符音高值标准化到有效的 MIDI 音高范围 \[0, 127] 内。

#### 参数

| 参数        | 类型          | 描述                      |
| --------- | ----------- | ----------------------- |
| `pitches` | `list[int]` | 原始音高序列，可能包含超出 MIDI 范围的值 |

#### 返回值

返回标准化后的音高序列，所有值都在 \[0, 127] 范围内。

---

## 注意事项

* 在使用 `--seed` 参数时，确保种子的随机性不会导致重复的结果，除非故意指定相同的种子。
* 温度参数影响生成的多样性，较高的温度会生成更多样的结果，较低的温度则使结果更为保守。
* `max_pitch_span_semitones` 参数有助于防止生成音符时出现过大的音高跨度，可以防止音域过度漂移。
