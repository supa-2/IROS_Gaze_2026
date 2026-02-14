"""
数据格式转换脚本 - 将 instruction-input-output 格式转换为 base 模型格式

支持多种微调框架格式：
1. JSONL (每行一个 JSON 对象)
2. Prompt-Completion 格式 (适合 OpenAI/简单的 fine-tuning)
3. Alpaca 格式 (instruction-based)
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import Counter


def create_prompt(instruction: str, input_text: str) -> str:
    """
    构造 prompt

    格式：
    < instruction > {instruction} < /instruction >
    < input > {input_text} < /input >
    < output >
    """
    if input_text and input_text.strip():
        prompt = f"<instruction>{instruction}</instruction>\n<input>{input_text}</input>\n<output>"
    else:
        prompt = f"<instruction>{instruction}</instruction>\n<output>"
    return prompt


def convert_to_base_format(
    input_path: str,
    output_dir: str,
    format_type: str = "jsonl",
    train_test_split: float = 0.9
) -> Dict:
    """
    转换数据格式

    Args:
        input_path: 输入文件路径 (cleaned JSON)
        output_dir: 输出目录
        format_type: 输出格式类型 (jsonl/alpaca)
        train_test_split: 训练集比例
    """
    print(f"[INFO] Loading data from: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[INFO] Total samples: {len(data)}")

    # 转换数据
    converted_data = []
    for sample in data:
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        output = sample['output']

        prompt = create_prompt(instruction, input_text)

        if format_type == "jsonl":
            # JSONL 格式: {"prompt": "...", "completion": "..."}
            converted_data.append({
                "prompt": prompt,
                "completion": output
            })
        elif format_type == "alpaca":
            # Alpaca 格式
            converted_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output
            })

    # 划分训练集和验证集
    split_idx = int(len(converted_data) * train_test_split)
    train_data = converted_data[:split_idx]
    val_data = converted_data[split_idx:]

    print(f"[INFO] Train set: {len(train_data)} samples")
    print(f"[INFO] Validation set: {len(val_data)} samples")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练集
    train_file = output_dir / f"train_{format_type}.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[INFO] Saved train set to: {train_file}")

    # 保存验证集
    val_file = output_dir / f"val_{format_type}.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[INFO] Saved validation set to: {val_file}")

    # 保存完整数据 (JSON 格式，方便查看)
    full_file = output_dir / f"full_{format_type}.json"
    with open(full_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved full data to: {full_file}")

    # 统计信息
    print("\n[INFO] Sample statistics:")
    print(f"  Average prompt length: {sum(len(item['prompt']) for item in train_data) / len(train_data):.0f} chars")
    print(f"  Average completion length: {sum(len(item['completion']) for item in train_data) / len(train_data):.0f} chars")

    # 显示示例
    print("\n[INFO] Example sample:")
    print("="*60)
    print("Prompt:")
    print(train_data[0]['prompt'][:200] + "...")
    print("\nCompletion:")
    print(train_data[0]['completion'][:100] + "...")
    print("="*60)

    return {
        'total_samples': len(converted_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'train_file': str(train_file),
        'val_file': str(val_file),
        'full_file': str(full_file)
    }


def convert_to_huggingface_format(
    input_path: str,
    output_dir: str,
    train_test_split: float = 0.9
):
    """
    转换为 HuggingFace Dataset 格式

    这种格式可以直接用于 transformers 的 Trainer
    """
    print(f"[INFO] Loading data from: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[INFO] Total samples: {len(data)}")

    # 构造文本对
    texts = []
    for sample in data:
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        output = sample['output']

        prompt = create_prompt(instruction, input_text)
        text = prompt + output

        texts.append({
            'text': text,
            'instruction': instruction,
            'input': input_text,
            'output': output
        })

    # 划分
    split_idx = int(len(texts) * train_test_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    print(f"[INFO] Train set: {len(train_texts)} samples")
    print(f"[INFO] Validation set: {len(val_texts)} samples")

    # 保存
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train_hf.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_texts, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved train set to: {train_file}")

    val_file = output_dir / "val_hf.json"
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_texts, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved validation set to: {val_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert gaze dataset to base model format')
    parser.add_argument('--input', type=str,
                       default=r'd:\Python\IROS 2026_gaze\IROS_AGENT\data\processed\gaze_cleaned.json',
                       help='Input cleaned JSON file')
    parser.add_argument('--output-dir', type=str,
                       default=r'd:\Python\IROS 2026_gaze\IROS_AGENT\data\processed\finetuning',
                       help='Output directory')
    parser.add_argument('--format', type=str, default='jsonl',
                       choices=['jsonl', 'alpaca', 'hf'],
                       help='Output format')
    parser.add_argument('--split', type=float, default=0.9,
                       help='Train/validation split ratio')

    args = parser.parse_args()

    print("="*60)
    print("Data Format Conversion for Base Model Fine-tuning")
    print("="*60)

    if args.format == 'hf':
        convert_to_huggingface_format(args.input, args.output_dir, args.split)
    else:
        stats = convert_to_base_format(
            args.input,
            args.output_dir,
            args.format,
            args.split
        )

    print("\n" + "="*60)
    print("[SUCCESS] Data conversion completed!")
    print("="*60)
