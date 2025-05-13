# MIDI 模型训练与验证说明文档

## 概述
本项目用于训练一个基于 Transformer 的模型，处理 MIDI 文件，预测音符序列。本文档将介绍如何使用该项目，相关类和函数的说明。

## 使用方法

1. **准备环境：**
   - 安装依赖库：`mido`, `torch`, `transformers`, `matplotlib`, `tqdm` 等。
   - 准备 MIDI 文件或 JSON 格式的音符数据。

2. **命令行训练：**
   你可以通过命令行运行以下脚本来训练模型：

   ```bash
   python train_model.py <num_epochs> <ckpt_path> -t <train_dataset> -v <val_dataset>
   ```

**参数说明：**

* `<num_epochs>`: 训练的总轮数。
* `<ckpt_path>`: 检查点保存路径。
* `-t <train_dataset>`: 训练集路径，可以指定多个数据集。
* `-v <val_dataset>`: 验证集路径（可选）。
* `-m <min-sequence-length>`: 最小序列长度，默认为 64。
* `-e <max-sequence-length>`: 最大序列长度，默认为 2^17。
* `-b <max-batch-size>`: 最大批次大小，默认为 8 \* 1536^2。
* `-l <learning-rate>`: 学习率，默认为 1e-2。
* `-w <weight-decay>`: 权重衰减系数，默认为 1e-2。
* `-n <num-heads>`: 多头注意力的头数量，默认为 12。
* `-d <dim-head>`: 注意力头的维度，默认为 64。
* `-f <dim-feedforward>`: 前馈层的维度，默认为 2048。
* `-s <num-layers>`: Transformer 编码器的层数，默认为 12。
* `-o <dropout>`: Dropout 概率，默认为 0.1。

## 类说明

### 1. **MidiDataset**

该类用于加载和处理 MIDI 文件，转换为模型可以使用的格式。

#### 参数

| 参数名                   | 类型                        | 作用                   |
| --------------------- | ------------------------- | -------------------- |
| `midi_dirs`           | `list[pathlib.Path]`      | 包含 MIDI/JSON 文件的目录列表 |
| `tokenizer`           | `PreTrainedTokenizerFast` | 用于音乐数据编码的分词器         |
| `min_sequence_length` | `int`                     | 最小序列长度（按音符表示）        |
| `max_sequence_length` | `int`                     | 最大序列长度（按乐谱表示）        |
| `show_progress`       | `bool`                    | 是否显示加载进度条            |

### 2. **MidiDatasetSampler**

该类用于从 MIDI 数据集中按批次生成索引。

#### 参数

| 参数名              | 类型            | 作用                           |
| ---------------- | ------------- | ---------------------------- |
| `dataset`        | `MidiDataset` | 包含 MIDI 数据的 `MidiDataset` 实例 |
| `max_batch_size` | `int`         | 每个批次的最大序列长度的平方和上限            |
| `drop_last`      | `bool`        | 是否丢弃最后一个批次（如果批次小于上限）         |

### 3. **train**

该函数用于训练模型，并返回每步的训练损失和困惑度。

#### 参数

| 参数名          | 类型                                                     | 作用                    |
| ------------ | ------------------------------------------------------ | --------------------- |
| `model`      | `MidiNet`                                              | 需要训练的神经网络模型           |
| `dataloader` | `DataLoader`                                           | 训练数据加载器               |
| `optimizer`  | `optim.Adam`                                           | 优化器                   |
| `criterion`  | `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]` | 损失函数（通常是交叉熵损失）        |
| `vocab_size` | `int`                                                  | 词汇表的大小（用于调整输出层的维度）    |
| `pad_token`  | `int`                                                  | 填充 token 的标记，用于忽略计算损失 |
| `device`     | `torch.device`                                         | 训练设备（`cuda` 或 `cpu`）  |
| `pbar_desc`  | `str`                                                  | 进度条描述（例如："训练"）        |

### 4. **validate**

该函数用于验证模型，并返回验证集上的平均损失和困惑度。

#### 参数

| 参数名          | 类型                                                     | 作用                    |
| ------------ | ------------------------------------------------------ | --------------------- |
| `model`      | `MidiNet`                                              | 需要验证的神经网络模型           |
| `dataloader` | `DataLoader`                                           | 验证数据加载器               |
| `criterion`  | `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]` | 损失函数（通常是交叉熵损失）        |
| `vocab_size` | `int`                                                  | 词汇表的大小（用于调整输出层的维度）    |
| `pad_token`  | `int`                                                  | 填充 token 的标记，用于忽略计算损失 |
| `device`     | `torch.device`                                         | 验证设备（`cuda` 或 `cpu`）  |
| `pbar_desc`  | `str`                                                  | 进度条描述（例如："验证"）        |

### 5. **plot\_training\_process**

该函数用于绘制训练过程中的损失和困惑度曲线。

#### 参数

| 参数名        | 类型                | 作用               |        |
| ---------- | ----------------- | ---------------- | ------ |
| `metrics`  | `dict[str, list]` | 包含训练和验证损失、困惑度的字典 |        |
| `img_path` | \`pathlib.Path    | str\`            | 图形保存路径 |

## 函数说明

### 1. **sequence\_collate\_fn**

该函数用于将多个样本合成统一长度的 batch。

#### 参数

| 参数名         | 类型                                        | 作用                    |
| ----------- | ----------------------------------------- | --------------------- |
| `batch`     | `list[tuple[torch.Tensor, torch.Tensor]]` | 包含多个样本的批次             |
| `pad_token` | `int`                                     | 填充 token 的标记，用于忽略计算损失 |

### 2. **empty\_cache**

该函数用于清理缓存，释放 GPU 内存。

---

## 结语

本项目提供了用于处理 MIDI 数据的类、方法和训练过程，可以训练一个音乐生成模型。你可以根据需要修改训练参数，来适配不同的数据集和模型配置。

