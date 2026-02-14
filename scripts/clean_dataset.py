"""
数据清洗脚本 - 移除不相关的样本
"""
import json
from pathlib import Path
from typing import List, Dict


def is_relevant_sample(sample: Dict) -> bool:
    """
    判断样本是否与视觉注意力预测相关

    规则：
    1. instruction 必须包含中文
    2. instruction 必须包含关键词：注意力、视觉、扫描、预测、热度
    3. 排除纯英文的通用指令
    """
    instruction = sample.get('instruction', '')

    # 关键词检查
    keywords = ['注意力', '视觉', '扫描', '预测', '热度', '注视', '等级', '展品', '显著性']
    has_chinese_keywords = any(kw in instruction for kw in keywords)

    # 排除英文指令
    is_english_task = instruction.encode('utf-8', errors='ignore').decode('utf-8') == instruction
    # 更准确的判断：检查是否包含中文字符
    has_chinese_char = any('\u4e00' <= char <= '\u9fff' for char in instruction)

    return has_chinese_char and has_chinese_keywords


def clean_dataset(input_path: str, output_path: str) -> Dict:
    """
    清洗数据集
    """
    print(f"[INFO] Reading data: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[INFO] Original samples: {len(data)}")

    # 过滤
    cleaned_data = [sample for sample in data if is_relevant_sample(sample)]

    print(f"[INFO] Cleaned samples: {len(cleaned_data)}")
    print(f"[INFO] Removed samples: {len(data) - len(cleaned_data)}")
    print(f"[INFO] Retention rate: {len(cleaned_data) / len(data) * 100:.1f}%")

    # 统计指令类型
    print("\n[INFO] Instruction type distribution:")
    type_counts = {}
    for sample in cleaned_data:
        instr = sample['instruction'][:60]
        type_counts[instr] = type_counts.get(instr, 0) + 1

    for instr, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {count:4d} : {instr}...")

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved cleaned data to: {output_path}")

    return {
        'original_count': len(data),
        'cleaned_count': len(cleaned_data),
        'removed_count': len(data) - len(cleaned_data),
        'type_distribution': type_counts
    }


if __name__ == '__main__':
    input_path = r'd:\Python\IROS 2026_gaze\IROS_AGENT\data\processed\gaze.json'
    output_path = r'd:\Python\IROS 2026_gaze\IROS_AGENT\data\processed\gaze_cleaned.json'

    stats = clean_dataset(input_path, output_path)

    print("\n" + "="*60)
    print("[SUCCESS] Data cleaning completed!")
    print("="*60)
