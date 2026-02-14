# Base 模型微调数据说明

## 数据处理流程

### 1. 数据清洗
- **输入**: `data/processed/gaze.json` (9158 条原始数据)
- **输出**: `data/processed/gaze_cleaned.json` (8159 条清洗后数据)
- **移除**: 999 条不相关样本（英文通用指令等）

### 2. 格式转换
为适配 base 模型微调，数据已转换为多种格式：

#### 文件列表

```
data/processed/finetuning/
├── train_jsonl.jsonl       # JSONL 格式训练集 (7343 条)
├── val_jsonl.jsonl         # JSONL 格式验证集 (816 条)
├── full_jsonl.json         # JSONL 完整数据 (8159 条)
├── train_hf.json           # HuggingFace 格式训练集
└── val_hf.json             # HuggingFace 格式验证集
```

#### 数据划分
- **训练集**: 7343 条 (90%)
- **验证集**: 816 条 (10%)
- **平均 prompt 长度**: 254 字符
- **平均 completion 长度**: 53 字符

## 数据格式

### JSONL 格式 (适合 OpenAI/LLaMA-Factory 等)

```json
{
  "prompt": "<instruction>规划一条包含注意力等级(A-E)的视觉扫描路径...</instruction>\n<input>当前场景可见展品...</input>\n<output>",
  "completion": "热度路径规划:\n1. [B] 展品名称\n2. [A] 展品名称..."
}
```

### HuggingFace 格式 (适合 Transformers Trainer)

```json
{
  "text": "<instruction>...</instruction>\n<input>...</input>\n<output>热度路径规划...",
  "instruction": "规划一条包含注意力等级(A-E)的视觉扫描路径...",
  "input": "当前场景可见展品...",
  "output": "热度路径规划:\n1. [B] ..."
}
```

## 推荐的 Base 模型

### Qwen 系列 (推荐)

```bash
# 模型选项
Qwen/Qwen2.5-0.5B    # 最小，测试用
Qwen/Qwen2.5-1.5B    # 小型，快速
Qwen/Qwen2.5-7B      # 推荐，性能和速度平衡
Qwen/Qwen2.5-14B     # 更强性能
Qwen/Qwen2.5-32B     # 最大，需要更多资源
```

**为什么选 Base 模型？**
- ✅ 没有对话模式干扰
- ✅ 更容易学习特定输出格式
- ✅ 微调效率更高
- ✅ 更不容易发生灾难性遗忘

### LLaMA 系列

```bash
meta-llama/Llama-3.2-3B-Instruct    # 3B 参数
meta-llama/Llama-3.2-1B-Instruct    # 1B 参数
```

## 微调建议

### 1. 使用 LLaMA-Factory (推荐)

LLaMA-Factory 支持多种格式，易于使用：

```bash
# 安装
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -r requirements.txt

# 配置数据集 (在 data/dataset_info.json 中添加)
{
  "gaze_attention": {
    "file_name": "train_jsonl.jsonl",
    "formatting": "prompt-completion",
    "columns": {
      "prompt": "prompt",
      "query": "",
      "response": "completion",
      "system": "",
      "history": []
    }
  }
}

# 运行微调
llamafactory-cli train config/qwen2_lora_sft.yaml
```

### 2. 使用 HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer

# 加载模型
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 加载数据
dataset = load_dataset("json", data_files={
    "train": "train_hf.json",
    "validation": "val_hf.json"
})

# 微调
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps"
    )
)

trainer.train()
```

### 3. 使用 PEFT/LoRA (节省显存)

```python
from peft import LoraConfig, get_peft_model

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## 微调参数建议

### Qwen2.5-7B Base

```yaml
learning_rate: 2e-5
num_train_epochs: 3-5
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
max_seq_length: 512
warmup_ratio: 0.1
lr_scheduler_type: cosine
weight_decay: 0.01

# LoRA 参数
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### 资源需求

| 模型 | 显存需求 (FP16) | 显存需求 (LoRA) | 推荐配置 |
|------|----------------|----------------|----------|
| Qwen2.5-1.5B | ~6 GB | ~3 GB | 单张 T4/GTX 1660 |
| Qwen2.5-7B | ~16 GB | ~8 GB | 单张 RTX 3090/4090 |
| Qwen2.5-14B | ~32 GB | ~16 GB | 单张 A6000/V100 |
| Qwen2.5-32B | ~64 GB | ~32 GB | 双张 A100 |

## 评估指标

微调后应评估：

1. **格式准确率**: 输出是否包含正确的前缀（"热度路径规划:"、"下一步预测:"）
2. **等级准确率**: A-E 等级预测是否正确
3. **序列准确率**: 完整路径的匹配度
4. **零样本泛化**: 在新场景上的表现

### 测试脚本

```python
# 测试推理
test_input = """
<instruction>基于当前视觉场景和历史注视行为，预测用户下一个关注目标的注意力等级 (A/B/C/D/E)。</instruction>
<input>当前场景可见展品(Visual Context):
- 【丁香花】: 一幅画着白色小花的画
- 【说明文字】: 展品介绍

历史行为:
1. [E] 说明文字
<output>
"""

inputs = tokenizer(test_input, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 数据统计

### 任务类型分布

1. **视觉扫描路径规划**: 3038 条 (37%)
   - 格式: "热度路径规划:\n1. [等级] 展品\n..."

2. **下一步预测**: 3038 条 (37%)
   - 格式: "下一步预测: [等级] 展品名称"

3. **特征归因分析**: 2083 条 (26%)
   - 格式: "特征归因: 目标【...】被标记为 X 级..."

### 注意力等级分布

- **A 级** (120秒): ~25%
- **B 级** (60秒): ~40%
- **C 级** (30秒): ~20%
- **D 级** (15秒): ~10%
- **E 级** (5秒): ~5%

## 常见问题

### Q1: 微调后输出格式不对？
A: Base 模型可能需要更多 epoch 学习格式。尝试：
- 增加 training epochs 到 5-10
- 提高 learning rate 到 5e-5
- 增加 format 相关的样本权重

### Q2: 模型输出了对话式前缀？
A: 这是 chat 模型的问题。使用 base 模型即可解决。

### Q3: 如何处理新展品？
A: 零样本泛化。测试时使用完全不同的展品名称，看模型是否能正确应用注意力等级。

### Q4: 显存不足？
A: 使用 LoRA + Gradient Checkpointing + 8bit 量化：
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
model.gradient_checkpointing_enable()
```

## 下一步

1. 选择合适的 base 模型（推荐 Qwen2.5-7B）
2. 选择微调框架（推荐 LLaMA-Factory）
3. 运行微调（3-5 epochs）
4. 评估效果
5. 如果效果好，可以尝试更大的模型（14B/32B）

---

*数据生成时间: 2025-02-05*
*适用任务: IROS 2026 - Eye-LLM 空间意图预测*
