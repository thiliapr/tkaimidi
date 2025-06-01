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
* `-v <val_dataset>`: 验证集路径。
* `-m <min-sequence-length>`: 最小序列长度
* `-e <max-sequence-length>`: 最大序列长度
* `-b <train-max-batch-size>`: 训练时，每个批次的序列长度的和上限。
* `-q <val-max-batch-size>`: 验证时，每个批次的序列长度的和上限。
* `-l <learning-rate>`: 学习率
* `-w <weight-decay>`: 权重衰减系数
* `-n <num-heads>`: 多头注意力的头数量
* `-d <dim-head>`: 注意力头的维度
* `-f <dim-feedforward>`: 前馈层的维度
* `-s <num-layers>`: Transformer 编码器的层数
* `-o <dropout>`: Dropout 概率

## 结语

本项目提供了用于处理 MIDI 数据的类、方法和训练过程，可以训练一个音乐生成模型。你可以根据需要修改训练参数，来适配不同的数据集和模型配置。

