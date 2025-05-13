# 训练分词器与数据转换模块
本模块提供MIDI数据处理和分词器训练功能，用于音乐生成模型的预处理流程。

## 功能概述
- MIDI数据到训练样本的转换
- 自定义BPE分词器训练
- 数据预处理和验证
- 命令行工具集成

## 核心功能
### 数据转换
#### `data_to_str(data: Iterator[int]) -> str`
将整数序列转换为字符串表示

##### 参数
- `data`: 整数序列(字符索引)

##### 返回
- 转换后的字符串

#### `str_to_data(string: str) -> list[int]`
将字符串转换回整数序列

##### 参数
- `string`: 编码后的字符串

##### 返回
- 原始整数序列

### 分词器训练
#### `train_sheet_tokenizer()`
训练专门处理音乐数据的BPE分词器

##### 参数
| 参数名 | 类型 | 说明 |
| - | - | - |
| model_data_samples | Iterator[str] | 训练样本迭代器 |
| vocab_size | int | 词汇表大小 |
| min_frequency | int | 最小词频阈值 |

##### 特点
- 支持特殊token([PAD],[UNK],[BOS],[EOS])
- 自动处理音乐符号编码
- 返回HuggingFace兼容的分词器

### 数据处理
#### `get_samples()`
从MIDI文件生成训练样本

##### 参数
| 参数名 | 类型 | 说明 |
| - | - | - |
| midi_dirs | list[pathlib.Path] | MIDI文件目录列表 |
| max_sequence_length | int | 最大序列长度 |
| show_progress | bool | 是否显示进度条 |

##### 流程
MIDI → 音符序列 → 电子乐谱 → 字符串编码

### 验证评估
#### `validate()`
评估分词器效果

##### 指标
- 总音符数
- 序列长度/音符数比例
- 词汇表使用率

#### `print_validation_results()`
格式化输出评估结果

## 命令行使用
```bash
python tokenizer.py ckpt -t train_data -t train_data_2 -v valid_data
```

### 参数说明
| 参数名 | 类型 | 说明 |
| - | - | - |
| ckpt_path | Path | 输出目录 |
| -t/--train-samples | list[Path] | 必须，训练数据目录，多选 |
| -v/--valid-samples | list[Path] | 验证数据目录，多选 |
| -s/--vocab-size | int | 词汇表大小 |
| -f/--min-frequency | int | 最小词频 |
| -m/--max-sequence-length | int | 最长允许多长的序列参与训练和测试（单位: 字符） |
| -y/--force | 选项 | 强制重新训练 |
