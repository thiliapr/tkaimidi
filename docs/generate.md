# TkAIMidi MIDI 生成器文档

`TkAIMidi` 是一个基于神经网络的 MIDI 音乐生成工具，支持从现有 MIDI 片段续写和自由创作。

## 核心功能

### 1. 音符序列生成

```python
def generate_midi(
    prompt: list[tuple[int, int]],
    model: MidiNet,
    tokenizer: PreTrainedTokenizerFast,
    seed: Optional[int] = None,
    temperature: float = 1.,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1,
    pitch_volatility_threshold: float = 20.,
    max_length: Optional[int] = None,
    device: torch.device = None
) -> Iterator[tuple[int, int]]
```

**参数说明**：
- `prompt`: 初始音符序列，格式为 `(音高, 间隔时间)` 元组列表
- `pitch_volatility_threshold`: 音高波动阈值（半音标准差），默认20
- `max_length`: 最大生成音符数量

**特性**：
- 实时音高稳定性控制
- 支持八度跳跃和调性变化
- 可中断的流式生成

### 2. MIDI轨道转换

```python
def notes_to_track(notes: list[int]) -> mido.MidiTrack
```

将音符序列转换为标准的MIDI轨道，包含：
- 音符开启/关闭事件
- 精确的时间控制（基于TIME_PRECISION）
- 自动添加结束标记

### 3. 音高处理工具

```python
def center_pitches(pitches: list[int]) -> list[tuple[int, int]]
def clamp_midi_pitch(pitches: list[int])
```

提供音高居中化和标准化功能，确保生成的MIDI符合规范。

## 使用示例

### 命令行接口

```bash
python generate.py ckpt output.mid
```

**参数说明**：
- `-t/--temperature`: 采样温度 (0.1-2.0)
- `-k/--top-k`: Top-K采样数量
- `-p/--max-pitch-span-semitones`: 最大允许音高跨度
- `-l/--max-length`: 最大生成音符数

### 编程接口

```python
if 1:
	import pathlib
	from generate import generate_midi
	from utils import notes_to_track
	from checkpoint import load_checkpoint
	from model import MidiNet

# 初始化模型和tokenizer
model = MidiNet(...)
tokenizer, state_dict = load_checkpoint(pathlib.Path("ckpt"), train=False)
model.load_state_dict(state_dict)

# 生成音乐
notes = list(generate_midi(
    prompt=[(0, 0)],
    model=model,
    tokenizer=tokenizer,
    temperature=0.7
))

# 保存为MIDI
track = notes_to_track(notes)
mido.MidiFile(tracks=[track]).save("output.mid")
```

## 技术架构

1. **模型结构**：
   - Transformer-based架构
   - 多头注意力机制
   - 可配置的层数和维度

2. **音乐表示**：
   - 将音符事件转换为token序列
   - 支持升降调、八度跳跃等控制事件

3. **生成控制**：
   - 温度采样
   - Top-K过滤
   - 重复惩罚机制

## 已知限制

1. 极端的温度参数可能导致不和谐的音乐
2. 生成的音乐结构依赖提示片段的质量
3. 复杂和弦的生成需要更大规模的模型
