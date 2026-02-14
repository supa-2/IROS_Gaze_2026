"""
数据集分析脚本 - 统计和可视化数据分布
"""
import json
from collections import Counter
from pathlib import Path


def analyze_attention_levels(data):
    """分析注意力等级分布"""
    level_counter = Counter()

    for sample in data:
        output = sample.get('output', '')
        # 统计 A-E 等级
        for level in ['A', 'B', 'C', 'D', 'E']:
            # 查找 [A], [B] 等模式
            count = output.count(f'[{level}]')
            level_counter[level] += count

    return level_counter


def analyze_instruction_types(data):
    """分析指令类型"""
    type_counter = Counter()

    for sample in data:
        instruction = sample['instruction']
        # 取前 60 个字符作为类型标识
        type_key = instruction[:60]
        type_counter[type_key] += 1

    return type_counter


def analyze_output_formats(data):
    """分析输出格式"""
    format_counter = Counter()

    for sample in data:
        output = sample['output']
        if output.startswith('热度路径规划'):
            format_counter['路径规划'] += 1
        elif output.startswith('下一步预测'):
            format_counter['下一步预测'] += 1
        elif output.startswith('特征归因'):
            format_counter['特征归因'] += 1
        else:
            format_counter['其他'] += 1

    return format_counter


def print_statistics(data_path: str):
    """打印数据统计信息"""
    print(f"[INFO] Loading data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")

    print(f"\n[Basic Info]")
    print(f"  Total samples: {len(data)}")

    # 统计文本长度
    prompt_lengths = []
    completion_lengths = []

    for sample in data:
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        output = sample['output']

        prompt_len = len(instruction) + len(input_text)
        completion_len = len(output)

        prompt_lengths.append(prompt_len)
        completion_lengths.append(completion_len)

    print(f"\n[Text Length Statistics]")
    print(f"  Prompt length:")
    print(f"    Min: {min(prompt_lengths)}")
    print(f"    Max: {max(prompt_lengths)}")
    print(f"    Avg: {sum(prompt_lengths) / len(prompt_lengths):.0f}")

    print(f"  Completion length:")
    print(f"    Min: {min(completion_lengths)}")
    print(f"    Max: {max(completion_lengths)}")
    print(f"    Avg: {sum(completion_lengths) / len(completion_lengths):.0f}")

    # 输出格式分布
    format_counter = analyze_output_formats(data)
    print(f"\n[Output Format Distribution]")
    for fmt, count in format_counter.most_common():
        percentage = count / len(data) * 100
        print(f"  {fmt}: {count} ({percentage:.1f}%)")

    # 注意力等级分布
    level_counter = analyze_attention_levels(data)
    total_mentions = sum(level_counter.values())
    print(f"\n[Attention Level Distribution]")
    print(f"  Total mentions: {total_mentions}")
    for level in ['A', 'B', 'C', 'D', 'E']:
        count = level_counter[level]
        percentage = count / total_mentions * 100 if total_mentions > 0 else 0
        print(f"  Level {level}: {count} ({percentage:.1f}%)")

    # 指令类型
    type_counter = analyze_instruction_types(data)
    print(f"\n[Top 10 Instruction Types]")
    for i, (instr, count) in enumerate(type_counter.most_common(10), 1):
        print(f"  {i}. {count:4d}条: {instr}...")

    # 显示示例
    print(f"\n[Sample Examples]")
    print("="*60)

    # 找不同类型的样本
    format_examples = {
        '路径规划': None,
        '下一步预测': None,
        '特征归因': None
    }

    for sample in data:
        output = sample['output']
        if output.startswith('热度路径规划') and format_examples['路径规划'] is None:
            format_examples['路径规划'] = sample
        elif output.startswith('下一步预测') and format_examples['下一步预测'] is None:
            format_examples['下一步预测'] = sample
        elif output.startswith('特征归因') and format_examples['特征归因'] is None:
            format_examples['特征归因'] = sample

        if all(v is not None for v in format_examples.values()):
            break

    for fmt, sample in format_examples.items():
        if sample:
            print(f"\n[{fmt}]")
            print(f"Instruction: {sample['instruction'][:80]}...")
            if sample.get('input'):
                print(f"Input: {sample['input'][:100]}...")
            print(f"Output: {sample['output'][:150]}...")

    print("\n" + "="*60)


if __name__ == '__main__':
    # 分析清洗后的数据
    cleaned_data_path = r'd:\Python\IROS 2026_gaze\IROS_AGENT\data\processed\gaze_cleaned.json'
    print_statistics(cleaned_data_path)
