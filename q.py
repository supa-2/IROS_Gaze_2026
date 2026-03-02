import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset

model_stub = "/home/g/models/qwen2.5-32b"
save_path = "/home/g/models/qwen2.5-32b-4bit-base"

# 显式指定 dtype 和 device
device = "cuda:0"  # 如果有足够显存，否则使用多卡映射
tokenizer = AutoTokenizer.from_pretrained(model_stub, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_stub,
    torch_dtype=torch.float16,
    device_map=device,  # 或 "auto" 但需配合内存参数
    trust_remote_code=True,
)

# 加载校准数据集（wikitext-2 作为示例）
calib_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# 可能需要预处理为模型输入的格式，例如 tokenize 并截断
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)
calib_data = calib_data.map(tokenize_fn, batched=True, remove_columns=["text"])

# 定义量化方案
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    sequential_update=True,
    block_size=128,           # 常用值
    damp_percent=0.01,         # 可选
)

# 执行量化
oneshot(
    model=model,
    tokenizer=tokenizer,
    recipe=recipe,
    output_dir=save_path,
    calibration_data=calib_data,   # 假设参数名如此，请查阅文档确认
    # 可能需要指定 batch_size 等
)

print(f"🎉 量化完成，模型保存至: {save_path}")