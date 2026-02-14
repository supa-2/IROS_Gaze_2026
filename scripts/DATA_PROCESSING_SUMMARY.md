# 数据处理完成总结

## 执行的操作

### 1. 数据清洗 ✅
- **脚本**: `scripts/clean_dataset.py`
- **输入**: `data/processed/gaze.json` (9158 条)
- **输出**: `data/processed/gaze_cleaned.json` (8159 条)
- **移除**: 999 条不相关样本（英文指令、通用任务等）

### 2. 格式转换 ✅
- **脚本**: `scripts/convert_to_base_format.py`
- **生成文件**:
  - `data/processed/finetuning/train_jsonl.jsonl` (7343 条训练集)
  - `data/processed/finetuning/val_jsonl.jsonl` (816 条验证集)
  - `data/processed/finetuning/train_hf.json` (HuggingFace 格式)
  - `data/processed/finetuning/val_hf.json` (HuggingFace 格式)

### 3. 数据分析 ✅
- **脚本**: `scripts/analyze_dataset.py`
- **统计报告**: 见下方

## 数据集统计

### 基本信息统计
```
总样本数: 8159 条
训练集:   7343 条 (90%)
验证集:   816 条  (10%)
```

### 文本长度统计
```
Prompt 长度:
  最小: 82 字符
  最大: 425 字符
  平均: 202 字符

Completion 长度:
  最小: 12 字符
  最大: 175 字符
  平均: 53 字符
```

### 任务类型分布
```
视觉扫描路径规划: 3038 条 (37.2%)
下一步预测:       3038 条 (37.2%)
特征归因分析:     2083 条 (25.5%)
```

### 注意力等级分布
```
Level A (120秒): 4461 次 (25.2%)
Level B (60秒):  7452 次 (42.0%)  ← 最多
Level C (30秒):  3288 次 (18.5%)
Level D (15秒):   836 次 (4.7%)
Level E (5秒):   1693 次 (9.5%)
```

## 数据格式

### Prompt 模板
```
<instruction>{指令}</instruction>
<input>{输入场景}</input>
<output>
```

### Output 示例

**类型 1: 路径规划**
```
热度路径规划:
1. [B] 丁香花
2. [B] 金鱼兰
3. [A] 说明文字
4. [A] 玉兰花开
```

**类型 2: 下一步预测**
```
下一步预测: [C] 鸟类
```

**类型 3: 特征归因**
```
特征归因: 目标【二十四节气圆盘】被标记为 B 级热点。
其对应的视觉显著性特征包括：融合虚拟现实技术的动态影像装置。
```

## 推荐的 Base 模型

### 最佳选择: Qwen2.5-7B-Base
```bash
模型名称: Qwen/Qwen2.5-7B
参数量:   7.6B
显存需求: ~16GB (FP16) / ~8GB (LoRA + INT8)
优势:     性能和速度平衡，中文能力强
```

### 其他选项
```bash
Qwen/Qwen2.5-1.5B    # 快速测试
Qwen/Qwen2.5-14B     # 更强性能
Qwen/Qwen2.5-32B     # 最高性能
```

## 微调建议

### 方案 1: LLaMA-Factory (最简单)
```bash
# 1. 安装
pip install llama-factory

# 2. 配置数据集 (data/dataset_info.json)
{
  "gaze_attention": {
    "file_name": "train_jsonl.jsonl",
    "formatting": "prompt-completion",
    "columns": {
      "prompt": "prompt",
      "response": "completion"
    }
  }
}

# 3. 运行微调
llamafactory-cli train your_config.yaml
```

### 方案 2: Transformers + PEFT (灵活)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 节省显存
    device_map="auto"
)

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./output/qwen-gaze",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
    ),
    peft_config=lora_config,
)

trainer.train()
```

### 关键参数
```yaml
learning_rate: 2e-5
num_train_epochs: 3-5
batch_size: 4
gradient_accumulation_steps: 2
max_seq_length: 512
warmup_ratio: 0.1

# LoRA
lora_r: 16
lora_alpha: 32
```

## 为什么 Base 模型更好？

### Chat 模型的问题
- ❌ 已学会对话模式（"我会帮你"、"让我分析"）
- ❌ 输出可能包含不必要的前缀
- ❌ 与特定格式任务冲突
- ❌ 容易发生灾难性遗忘

### Base 模型的优势
- ✅ 没有对话习惯干扰
- ✅ 更容易学习特定格式
- ✅ 微调效率更高
- ✅ 直接输出结果，没有多余内容

## 下一步操作

1. **选择模型**: 推荐从 Qwen2.5-7B-Base 开始
2. **选择框架**: 推荐使用 LLaMA-Factory（最简单）
3. **运行微调**: 3-5 个 epochs
4. **评估效果**: 检查格式、准确率、泛化能力
5. **如果效果好**: 尝试更大的模型（14B/32B）

## 测试推理

微调完成后，测试推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./output/qwen-gaze"  # 你的微调模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

test_prompt = """<instruction>规划一条包含注意力等级(A-E)的视觉扫描路径，用于构建注意力热图。</instruction>
<input>当前场景可见展品(Visual Context):
- 【丁香花】: 一幅画着白色小花的画
- 【说明文字】: 展品介绍
- 【金鱼兰】: 一幅画着金鱼状花朵的画
<output>"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
# 期望输出:
# 热度路径规划:
# 1. [B] 丁香花
# 2. [A] 说明文字
# 3. [B] 金鱼兰
```

## 文件清单

### 新创建的脚本
```
scripts/
├── clean_dataset.py           # 数据清洗
├── convert_to_base_format.py  # 格式转换
└── analyze_dataset.py         # 数据分析
```

### 处理后的数据
```
data/processed/
├── gaze.json                  # 原始数据 (9158 条)
├── gaze_cleaned.json          # 清洗后数据 (8159 条)
└── finetuning/
    ├── train_jsonl.jsonl      # JSONL 训练集 (7343 条)
    ├── val_jsonl.jsonl        # JSONL 验证集 (816 条)
    ├── full_jsonl.json        # JSONL 完整数据
    ├── train_hf.json          # HF 格式训练集
    ├── val_hf.json            # HF 格式验证集
    └── README.md              # 详细说明文档
```

## 常见问题

**Q: 微调后模型还是输出对话式内容？**
A: 确认使用的是 base 模型，不是 chat 模型。

**Q: 格式不对，缺少前缀？**
A: 增加 training epochs，或提高 format 样本的学习率。

**Q: 显存不足？**
A: 使用 LoRA + 8bit 量化 + gradient checkpointing。

**Q: 效果不好？**
A: 检查数据质量、调整超参数、尝试更大的模型。

---

*处理完成时间: 2025-02-05*
*数据用途: IROS 2026 Eye-LLM 空间意图预测系统*
