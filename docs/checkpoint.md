# 模型检查点管理文档

## 功能概述
提供模型训练过程中的检查点保存与加载功能，支持：
- 完整训练状态保存（模型参数+优化器状态+训练指标）
- 灵活加载模式（仅推理模式/完整训练恢复）
- 自动处理分布式训练场景
- 训练指标持久化

## 核心函数
### `save_checkpoint` 函数
#### 功能
保存完整的训练状态快照

#### 参数说明
| 参数 | 类型 | 说明 |
| - | - | - |
| `model` | MidiNet | 要保存的模型实例 |
| `optimizer_state_dict` | AdamW | 优化器状态字典 |
| `metrics` | dict[str, Any] | 训练指标字典 |
| `path` | Path | 保存路径 |

#### 保存内容
1. **模型参数**
   - 自动处理DataParallel封装
   - 保存为`model.pth`
2. **优化器状态**
   - 保存为`optimizer.pth`
3. **训练指标**
   - 包括验证集/训练集的loss
   - 保存为`metrics.json`

### `load_checkpoint` 函数
#### 功能
加载检查点恢复训练状态

#### 参数说明
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | Path | 必填 | 检查点目录 |
| `train` | bool | False | 是否加载训练状态 |

#### 返回内容
根据`train`参数返回不同内容：
- **False** (推理模式):
  ```python
  return tokenizer, model_state
  ```
- **True** (训练恢复模式):
  ```python
  return tokenizer, model_state, optimizer_state, metrics
  ```

#### 文件加载优先级
1. 必须存在`tokenizer`目录
2. 可选加载项：
   - `model.pth` (模型参数)
   - `optimizer.pth` (优化器状态)
   - `metrics.json` (训练指标)

## 使用示例
### 保存检查点
```python
save_checkpoint(
    model=training_model,
    optimizer_state_dict=optimizer,
    metrics={
        "val_ppl": [...],
        "train_ppl": [...]
    },
    path=Path("checkpoints/epoch_10")
)
```

### 加载检查点
#### 恢复训练
```python
tokenizer, model_state, optimizer_state, metrics = load_checkpoint(
    path=Path("checkpoints/epoch_10"),
    train=True
)
```
#### 仅推理
```python
tokenizer, model_state = load_checkpoint(
    path=Path("checkpoints/epoch_10")
)
```

## 文件结构规范
```
checkpoint_dir/
├── tokenizer/        # 分词器文件
├── model.pth        # 模型参数
├── optimizer.pth    # 优化器状态
└── metrics.json     # 训练指标
```
