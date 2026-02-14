# ShareGPT + JSON 结构化格式 - LLaMA-Factory 微调指南

## 数据格式说明

### 格式特点

- ✅ **ShareGPT 对话格式**：LLaMA-Factory 原生支持
- ✅ **JSON 结构化**：输入输出都是 JSON，像代码一样清晰
- ✅ **自动关键词提取**：区分重复名称（如：说明文字（墙面展签））
- ✅ **保留完整特征**：features 字段保留所有视觉描述

---

## 数据示例

### 输入格式（Human）

```json
{
  "task": "plan_scan_path",
  "exhibits": [
    {
      "name": "丁香花",
      "features": "一幅画着由白色圆盆栽开满白色小花，并且绿叶繁盛的画，直立挂起来，背景为黑色"
    },
    {
      "name": "说明文字（墙面展签）",
      "features": "墙面展签，印有展品名称"极乐鸟"，无长文本。"
    },
    {
      "name": "金鱼兰",
      "features": "一幅画着土红色盆子载种着一支叶片细长，花朵呈金鱼状的画，该画挂在展厅中央"
    }
  ]
}
```

### 输出格式（GPT）

```json
{
  "scan_path": [
    {
      "name": "丁香花",
      "attention_level": "B"
    },
    {
      "name": "说明文字（墙面展签）",
      "attention_level": "A"
    },
    {
      "name": "金鱼兰",
      "attention_level": "B"
    }
  ]
}
```

---

## 数据统计

| 指标 | 数值 |
|------|------|
| 总样本数 | 8159 条 |
| 训练集 | 7343 条 (90%) |
| 验证集 | 816 条 (10%) |
| 任务类型 | 3 种 |

### 任务类型

1. **plan_scan_path**（路径规划）：3038 条
2. **predict_next**（下一步预测）：3038 条
3. **其他任务**：2083 条

---

## 关键词提取规则

自动提取关键词来区分重复的展品名称：

### 规则 1：提取引号中的内容
```json
{
  "name": "说明文字",
  "features": "墙面展签，印有展品名称"极乐鸟"，无长文本。"
}
// ↓ 提取关键词
{
  "name": "说明文字（极乐鸟）",
  "features": "墙面展签，印有展品名称"极乐鸟"，无长文本。"
}
```

### 规则 2：提取主题词
```json
{
  "name": "说明文字",
  "features": "四季花，"摒弃传统的写实，以极致的概括与提炼...""
}
// ↓ 提取关键词
{
  "name": "说明文字（四季花）",
  "features": "..."
}
```

### 规则 3：描述性关键词
```json
{
  "name": "人",
  "features": "桌子上放着一幅用素描画着抽烟的人，旁边还有一个影子"
}
// ↓ 提取关键词
{
  "name": "人（抽烟）",
  "features": "..."
}
```

---

## 文件结构

```
data/processed/sharegpt_json/
├── train_sharegpt.jsonl       # 训练集（7343条）
├── val_sharegpt.jsonl         # 验证集（816条）
├── full_sharegpt.json         # 完整数据（JSON格式）
├── dataset_info.json          # LLaMA-Factory 数据集配置
└── README.md                  # 本文件
```

---

## LLaMA-Factory 使用方法

### 1. 配置数据集

将 `dataset_info.json` 的内容添加到 LLaMA-Factory 的配置文件中：

**文件位置**：`LLaMA-Factory/data/dataset_info.json`

**添加内容**：
```json
{
  "gaze_attention_json": {
    "file_name": "train_sharegpt.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "roles": {
        "human": "from",
        "gpt": "from"
      },
      "content": "value"
    }
  }
}
```

### 2. 复制数据文件

```bash
# 复制训练数据到 LLaMA-Factory/data/
cp data/processed/sharegpt_json/train_sharegpt.jsonl /path/to/LLaMA-Factory/data/
cp data/processed/sharegpt_json/val_sharegpt.jsonl /path/to/LLaMA-Factory/data/
```

### 3. 创建微调配置

**文件**：`qwen2_lora_sft_sharegpt.yaml`

```yaml
### Model
model_name_or_path: Qwen/Qwen2.5-7B  # 使用 Base 模型

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32

### Dataset
dataset: gaze_attention_json
dataset_dir: data
template: qwen
cutoff_len: 1024  # 增加到1024，因为JSON格式较长
max_samples: 100000
overwrite_cache: true

### Output
output_dir: saves/qwen2-7b-gaze-json
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### Train
per_device_train_batch_size: 2  # 减小batch size，因为JSON更长
gradient_accumulation_steps: 4
learning_rate: 5.0e-05
num_train_epochs: 5
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### Eval
val_size: 0.1
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 100
```

### 4. 运行微调

```bash
cd LLaMA-Factory

llamafactory-cli train qwen2_lora_sft_sharegpt.yaml
```

或使用命令行：

```bash
llamafactory-cli train \
  --model_name Qwen/Qwen2.5-7B \
  --stage sft \
  --dataset gaze_attention_json \
  --dataset_dir data \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 16 \
  --output_dir saves/qwen2-7b-gaze-json \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 5 \
  --learning_rate 5e-05 \
  --cutoff_len 1024
```

---

## 推荐配置

### 不同模型大小

| 模型 | 参数量 | Batch Size | 显存需求 | 训练时间 |
|------|--------|-----------|---------|---------|
| Qwen2.5-1.5B | 1.5B | 4 | ~4GB | ~30分钟 |
| Qwen2.5-7B | 7B | 2 | ~8GB | ~2-3小时 |
| Qwen2.5-14B | 14B | 1 | ~16GB | ~5-6小时 |
| Qwen2.5-32B | 32B | 1 | ~32GB | ~10-12小时 |

**注意**：因为 JSON 格式比纯文本长，建议：
- 减小 batch_size
- 增加 cutoff_len 到 1024 或 2048
- 使用 gradient_accumulation_steps 补偿

---

## 推理测试

### 命令行推理

```bash
llamafactory-cli chat \
  --model_name Qwen/Qwen2.5-7B \
  --adapter_name_or_path saves/qwen2-7b-gaze-json \
  --template qwen
```

### 测试输入示例

```json
{
  "task": "plan_scan_path",
  "exhibits": [
    {
      "name": "丁香花",
      "features": "一幅画着由白色圆盆栽开满白色小花，并且绿叶繁盛的画，直立挂起来，背景为黑色"
    },
    {
      "name": "金鱼兰",
      "features": "一幅画着土红色盆子载种着一支叶片细长，花朵呈金鱼状的画，该画挂在展厅中央"
    }
  ]
}
```

### 期望输出

```json
{
  "scan_path": [
    {
      "name": "丁香花",
      "attention_level": "B"
    },
    {
      "name": "金鱼兰",
      "attention_level": "B"
    }
  ]
}
```

---

## 格式对比

### vs Alpaca 格式

| 对比项 | Alpaca | ShareGPT + JSON |
|--------|--------|-----------------|
| 结构化程度 | 较低 | **高** ✓ |
| base 模型学习难度 | 较难 | **容易** ✓ |
| 格式明确性 | 一般 | **非常明确** ✓ |
| token 效率 | 较高 | 较低（JSON 开销） |
| 调试友好度 | 一般 | **好** ✓ |

### vs 纯文本格式

| 对比项 | 纯文本 | JSON 结构化 |
|--------|--------|------------|
| 模糊性 | **高** ✗ | 低 ✓ |
| 机器友好 | 一般 | **高** ✓ |
| 可扩展性 | 差 | **好** ✓ |
| 人类可读性 | **高** ✓ | 中等 |

---

## 常见问题

### Q1: 显存不足？

**A**: 使用以下参数优化：
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
cutoff_len: 512  # 减小序列长度
# 或使用 8bit 量化
--load_in_8bit true
```

### Q2: 训练太慢？

**A**: JSON 格式确实比纯文本长，可以：
- 减小 `cutoff_len`
- 使用更小的模型（1.5B 测试）
- 减少 `num_train_epochs` 到 3

### Q3: 输出格式不对？

**A**: 确认：
1. 使用的是 Base 模型（不是 Chat）
2. 数据格式正确（JSON 结构完整）
3. 增加训练轮数

### Q4: 如何处理新展品？

**A**:
- 零样本推理：直接输入新的 JSON
- 模型应该能泛化到未见过的展品
- 确保新展品的 features 描述清晰

---

## 优势总结

✅ **结构化清晰**：JSON 格式，输入输出一目了然
✅ **易于学习**：base 模型更容易学习结构化模式
✅ **无歧义性**：明确的字段定义，减少理解偏差
✅ **可扩展性**：未来可添加更多字段（如位置、类型等）
✅ **调试友好**：JSON 格式便于检查和调试

---

## 下一步

1. ✅ 数据已准备完成
2. 📌 安装 LLaMA-Factory
3. 📌 配置数据集（dataset_info.json）
4. 📌 复制数据文件
5. 📌 运行微调
6. 📌 测试推理效果

---

*数据生成时间: 2025-02-05*
*格式: ShareGPT + JSON 结构化*
*适用: Qwen2.5 Base 系列模型*
*框架: LLaMA-Factory*
