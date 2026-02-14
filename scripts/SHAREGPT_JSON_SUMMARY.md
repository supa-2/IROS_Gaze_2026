# ✅ ShareGPT + JSON 结构化格式转换完成

## 处理结果

| 指标 | 数值 |
|------|------|
| 总样本数 | 8159 条 |
| 训练集 | 7343 条 (90%) |
| 验证集 | 816 条 (10%) |
| 转换成功率 | 100% ✓ |

---

## 数据格式

### ShareGPT + JSON 结构化

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "```json\n{\n  \"task\": \"plan_scan_path\",\n  \"exhibits\": [\n    {\"name\": \"丁香花\", \"features\": \"...\"},\n    {\"name\": \"说明文字（墙面展签）\", \"features\": \"...\"}\n  ]\n}\n```"
    },
    {
      "from": "gpt",
      "value": "```json\n{\n  \"scan_path\": [\n    {\"name\": \"丁香花\", \"attention_level\": \"B\"},\n    {\"name\": \"说明文字（墙面展签）\", \"attention_level\": \"A\"}\n  ]\n}\n```"
    }
  ]
}
```

---

## 关键特性

### ✅ 自动关键词提取

区分重复的展品名称：

| 原始名称 | 描述 | 处理后 |
|---------|------|--------|
| 说明文字 | 墙面展签，印有展品名称"极乐鸟" | 说明文字（极乐鸟） |
| 说明文字 | 四季花，"摒弃传统的写实..." | 说明文字（四季花） |
| 人 | 桌子上放着一幅用素描画着抽烟的人 | 人（抽烟） |
| 人 | 桌子上放着一幅用素描画着一个盘腿坐着的人 | 人（盘腿坐） |

### ✅ 名称映射

输入输出中的名称完全一致：

**输入**：
```json
{"name": "说明文字（极乐鸟）", "features": "..."}
```

**输出**：
```json
{"name": "说明文字（极乐鸟）", "attention_level": "A"}
```

---

## 文件位置

```
data/processed/sharegpt_json/
├── train_sharegpt.jsonl       # ✅ 训练集（7343条）
├── val_sharegpt.jsonl         # ✅ 验证集（816条）
├── full_sharegpt.json         # ✅ 完整数据（JSON格式）
├── dataset_info.json          # ✅ LLaMA-Factory 配置
└── README.md                  # ✅ 详细说明
```

---

## 快速开始（3步）

### 1️⃣ 配置数据集

编辑 `LLaMA-Factory/data/dataset_info.json`，添加：

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
      }
    }
  }
}
```

### 2️⃣ 复制数据文件

```bash
cp data/processed/sharegpt_json/train_sharegpt.jsonl LLaMA-Factory/data/
cp data/processed/sharegpt_json/val_sharegpt.jsonl LLaMA-Factory/data/
```

### 3️⃣ 开始微调

```bash
cd LLaMA-Factory

llamafactory-cli train \
  --model_name Qwen/Qwen2.5-7B \
  --stage sft \
  --dataset gaze_attention_json \
  --finetuning_type lora \
  --lora_rank 16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 5 \
  --learning_rate 5e-05 \
  --cutoff_len 1024
```

---

## 推荐配置

| 模型 | Batch Size | Gradient Accum | 显存 | 时间 |
|------|-----------|---------------|------|------|
| Qwen2.5-1.5B | 4 | 2 | ~4GB | ~30分钟 |
| Qwen2.5-7B | 2 | 4 | ~8GB | ~2-3小时 |
| Qwen2.5-14B | 1 | 8 | ~16GB | ~5-6小时 |

**注意**：JSON 格式比纯文本长，需要更小的 batch size

---

## 优势总结

✅ **结构化清晰** - JSON 格式，像代码一样
✅ **易于学习** - Base 模型更容易理解
✅ **无歧义性** - 明确的字段定义
✅ **自动去重** - 关键词自动提取
✅ **格式一致** - 输入输出名称完全匹配

---

## 脚本文件

```
scripts/
└── convert_to_sharegpt_json.py  # 转换脚本
```

**功能**：
- 自动提取关键词区分重复名称
- 保留完整的 features 描述
- ShareGPT 格式输出
- 名称映射确保一致性

---

## 下一步

1. ✅ 数据已准备
2. 📌 安装 LLaMA-Factory
3. 📌 配置数据集
4. 📌 复制数据文件
5. 📌 运行微调
6. 📌 测试效果

---

*完成时间: 2025-02-05*
*格式: ShareGPT + JSON 结构化*
*适用: Qwen2.5 Base 模型 + LLaMA-Factory*
