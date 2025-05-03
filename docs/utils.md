# 训练和推理工具集
本模块提供MIDI处理、数据规范化和音高标准化等功能，用于音乐生成模型的训练和推理。

## 功能概述
- MIDI文件到音符序列的转换
- 音符序列到电子乐谱的编码
- 电子乐谱到音符序列的解码
- 显存管理工具

## 模块函数
### `midi_to_notes(midi_file: mido.MidiFile) -> list[tuple[int, int]]`
从MIDI文件中提取音符信息并返回音符及其相对时间间隔的列表。

#### 参数:
- `midi_file`: 要提取的MIDI文件对象

#### 返回:
- 包含音符和相对时间间隔的列表，格式为`(音高, 时间间隔)`

#### 特点:
- 自动跳过打击乐通道
- 处理音色变化事件
- 时间间隔压缩和规范化
- 去除重复音符

#### 示例:
```python
midi_file = mido.MidiFile("example.mid")
notes = midi_to_notes(midi_file)
```

### `notes_to_sheet(notes: list[tuple[int, int]], lookahead_count: int) -> tuple[list[tuple[str, int]], list[int]]`
将音符列表转换为电子乐谱编码。

#### 参数:
- `notes`: MIDI音符列表，格式为`(音高, 时间间隔)`
- `lookahead_count`: 调整音高时检查后续音符的数量

#### 返回:
- `sheet`: 电子乐谱事件列表
- `positions`: 每个音符在乐谱中的位置索引

#### 编码规则:
- 0-11: 音符(音阶中的音高)
- 12: 音高下调半音
- 13: 音高上调半音
- 14: 音符下跳八度
- 15: 音符上跳八度
- 16: 时间间隔

### `sheet_to_notes(sheet: list[int]) -> list[tuple[int, int]]`
将电子乐谱解码为音符列表。

#### 参数:
- `sheet`: 由`notes_to_sheet()`生成的电子乐谱

#### 返回:
- MIDI音符列表，格式为`(音高, 时间间隔)`

#### 特点:
- 自动处理全局音高偏移
- 处理八度跳跃
- 规范化最小音高为0

### `empty_cache()`
清空CUDA显存缓存并执行垃圾回收。

#### 特点:
- 自动检测CUDA可用性
- 同时执行Python垃圾回收

## 使用示例
```python
# 完整工作流程示例
midi_file = mido.MidiFile("input.mid")
notes = midi_to_notes(midi_file)
sheet, positions = notes_to_sheet(notes)
reconstructed_notes = sheet_to_notes(sheet)

# 显存清理
empty_cache()
```
