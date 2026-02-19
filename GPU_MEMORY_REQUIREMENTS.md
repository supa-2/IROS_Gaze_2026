# 模型显存需求参考表

## 概述

本文档列出了项目中使用的各种模型的显存（VRAM）需求，包括推理和训练场景。

---

## SAM 2 模型 (图像分割)

| 模型版本 | 参数量 | 模型大小 | 推理显存 (FP16) | 训练显存 (FP16) | 推荐配置 |
|----------|--------|----------|------------------|-----------------|----------|
| **sam2-hiera-tiny** | ~20M | ~40MB | 2 GB | 6 GB | GTX 1060 / RTX 2060 |
| **sam2-hiera-small** | ~40M | ~140MB | 4 GB | 10 GB | RTX 3060 / RTX 4060 |
| **sam2-hiera-base** | ~100M | ~360MB | 8 GB | 18 GB | RTX 3070 / RTX 4070 |
| **sam2-hiera-large** | ~224M | ~2.4GB | 16 GB | 32 GB | RTX 3090 / RTX 4090 / A6000 |

**注意**：
- FP32 格式需要约 2x 显存
- Batch size > 1 会线性增加显存需求
- 实际显存占用还受输入图片分辨率影响

---

## Qwen 系列模型 (LLM)

### 推理显存需求

| 模型 | 参数量 | 量化 | 推理显存 | 推荐配置 |
|------|--------|------|----------|----------|
| **Qwen2.5-0.5B** | 0.5B | FP16 | 1 GB | GTX 1060 / CPU |
| **Qwen2.5-0.5B** | 0.5B | FP32 | 2 GB | GTX 1060 / CPU |
| **Qwen2.5-1.5B** | 1.5B | FP16 | 3 GB | GTX 1060 / RTX 2060 |
| **Qwen2.5-1.5B** | 1.5B | 8-bit | 1.5 GB | GTX 1060 |
| **Qwen2.5-3B** | 3B | FP16 | 6 GB | RTX 3060 / RTX 4060 |
| **Qwen2.5-3B** | 3B | 8-bit | 3 GB | RTX 3060 / RTX 4060 |
| **Qwen2.5-7B** | 7B | FP16 | 14 GB | RTX 3080 / RTX 4080 |
| **Qwen2.5-7B** | 7B | 8-bit | 7 GB | RTX 3060 / RTX 4060 |
| **Qwen2.5-7B** | 7B | 4-bit | 4 GB | GTX 1060 / RTX 2060 |
| **Qwen2.5-14B** | 14B | FP16 | 28 GB | A6000 / V100 |
| **Qwen2.5-14B** | 14B | 8-bit | 14 GB | RTX 3090 / RTX 4090 |
| **Qwen2.5-14B** | 14B | 4-bit | 8 GB | RTX 3060 Ti / RTX 4070 |
| **Qwen2.5-32B** | 32B | FP16 | 64 GB | 2x A100 |
| **Qwen2.5-32B** | 32B | 8-bit | 32 GB | A100 / RTX 6000 Ada |
| **Qwen2.5-32B** | 32B | 4-bit | 16 GB | RTX 3090 / RTX 4090 |
| **Qwen2-VL-7B** | 7B | FP16 | 18 GB | RTX 3080 / RTX 4080 |

### LoRA 微调显存需求

| 模型 | 微调方式 | Batch Size | 显存需求 | 推荐配置 |
|------|----------|------------|----------|----------|
| **Qwen2.5-0.5B** | LoRA | 4 | ~4 GB | GTX 1060 |
| **Qwen2.5-0.5B** | LoRA | 8 | ~6 GB | RTX 2060 |
| **Qwen2.5-1.5B** | LoRA | 4 | ~6 GB | RTX 2060 / RTX 3060 |
| **Qwen2.5-1.5B** | LoRA | 8 | ~10 GB | RTX 3060 |
| **Qwen2.5-7B** | LoRA | 1 | ~8 GB | RTX 3060 / RTX 4060 |
| **Qwen2.5-7B** | LoRA | 2 | ~12 GB | RTX 3070 / RTX 4070 |
| **Qwen2.5-7B** | LoRA | 4 | ~16 GB | RTX 3080 / RTX 4080 |
| **Qwen2.5-14B** | LoRA | 1 | ~16 GB | RTX 3080 / RTX 4080 |
| **Qwen2.5-14B** | LoRA | 2 | ~24 GB | RTX 3090 / RTX 4090 |
| **Qwen2.5-32B** | LoRA | 1 | ~32 GB | A6000 / V100 |

### 全参数微调显存需求

| 模型 | Batch Size | 显存需求 (FP16) | 推荐配置 |
|------|------------|------------------|----------|
| **Qwen2.5-1.5B** | 1 | ~12 GB | RTX 3060 |
| **Qwen2.5-1.5B** | 4 | ~20 GB | RTX 3070 / RTX 4070 |
| **Qwen2.5-7B** | 1 | ~24 GB | RTX 3090 / RTX 4090 |
| **Qwen2.5-7B** | 2 | ~40 GB | A6000 |
| **Qwen2.5-14B** | 1 | ~48 GB | A6000 / 2x RTX 3090 |

---

## 其他常用模型

| 模型 | 类型 | 参数量 | 推理显存 (FP16) | 训练显存 (LoRA) |
|------|------|--------|------------------|------------------|
| **Llama-3.2-1B** | LLM | 1B | 2 GB | 4 GB |
| **Llama-3.2-3B** | LLM | 3B | 6 GB | 10 GB |
| **GPT-4o-mini** | API | - | - | - (云端) |
| **CLIP-ViT-B** | 视觉编码器 | - | 2 GB | - |

---

## GPU 配置参考

| GPU | 显存 | Turing (RTX 20系) | Ampere (RTX 30系) | Ada (RTX 40系) |
|-----|------|-------------------|--------------------|------------------|
| GTX 1060 | 6 GB | ❌ 不支持 Tensor Core | - | - |
| RTX 2060 | 6 GB | ✅ 支持 | - | - |
| RTX 3060 | 12 GB | - | ✅ 推荐 | - |
| RTX 3060 Ti | 8 GB | - | ✅ 推荐 | - |
| RTX 3070 | 8 GB | - | ✅ 推荐 | - |
| RTX 3080 | 10 GB | - | ✅ 推荐 | - |
| RTX 3080 Ti | 12 GB | - | ✅ 推荐 | - |
| RTX 3090 | 24 GB | - | ✅ 最佳 | - |
| RTX 4060 | 8 GB | - | - | ✅ 推荐 |
| RTX 4070 | 12 GB | - | - | ✅ 推荐 |
| RTX 4080 | 16 GB | - | - | ✅ 推荐 |
| RTX 4090 | 24 GB | - | - | ✅ 最佳 |
| A6000 | 48 GB | - | - | - | 专业级 |
| A100 | 40/80 GB | - | - | - | 专业级 |
| V100 | 16/32 GB | - | - | - | 专业级 |

---

## 显存优化技巧

### 1. 量化 (Quantization)

| 量化方式 | 显存节省 | 精度损失 |
|----------|----------|----------|
| FP32 → FP16 | 50% | 几乎无 |
| FP16 → 8-bit | 50% | 轻微 |
| FP16 → 4-bit | 75% | 中等 |

使用方法：
```python
# 8-bit 量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit 量化 (NF4)
from bitsandbytes import quantization_config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config(load_in_4bit=True),
    device_map="auto"
)
```

### 2. Gradient Checkpointing

```python
# 在训练时启用，节省 ~30% 显存，但训练慢约 20%
model.gradient_checkpointing_enable()
```

### 3. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
```

### 4. 梯度累积 (Gradient Accumulation)

```python
# 模拟更大 batch size，但不增加显存
gradient_accumulation_steps = 4
effective_batch_size = per_device_batch_size * gradient_accumulation_steps
```

### 5. DeepSpeed ZeRO

```bash
# 可处理超过单卡显存的模型
deepspeed --num_gpus=1 your_script.py --deepspeed_config ds_config.json
```

---

## 项目推荐配置

### 低配置 (8GB 显存)

```yaml
模型: Qwen2.5-1.5B
量化: 8-bit
Batch Size: 2
Gradient Accumulation: 4
预估显存: ~6 GB
```

### 中配置 (12GB 显存)

```yaml
模型: Qwen2.5-7B
量化: 4-bit 或 8-bit
Batch Size: 2
Gradient Accumulation: 4
预估显存: ~8 GB
```

### 高配置 (24GB 显存)

```yaml
模型: Qwen2.5-14B
量化: FP16
Batch Size: 4
Gradient Accumulation: 2
预估显存: ~16 GB
```

### 专业配置 (48GB+ 显存)

```yaml
模型: Qwen2.5-32B
量化: FP16
Batch Size: 8
预估显存: ~32 GB
```

---

## 显存监控命令

```bash
# 实时监控 GPU 显存
watch -n 1 nvidia-smi

# 查看显存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Python 中监控
import torch
print(f"已用显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"预留显存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

---

*最后更新: 2025-02-14*
