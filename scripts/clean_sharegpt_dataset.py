#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ShareGPT数据集清洗脚本

问题：sharegpt_json中的features字段包含大量LLM生成的填充文本
解决方案：从原始finetuning数据中提取真实描述，替换填充文本
"""

import json
import re
from pathlib import Path
from typing import Dict


def extract_exhibits_from_original(prompt: str) -> Dict[str, str]:
    """
    从原始数据的prompt中提取展品描述

    格式：【展品名】: 描述内容
    """
    exhibits = {}

    # 匹配 【展品名】: 描述
    pattern = r'- 【(.+?)】:\s*(.+?)(?=\n|$)'
    matches = re.findall(pattern, prompt)

    for name, features in matches:
        # 清理前后空格和特殊字符
        name = name.strip()
        features = features.strip()
        exhibits[name] = features

    return exhibits


def clean_filling_text(features: str) -> str:
    """
    移除LLM生成的填充文本

    填充文本模式：
    - "这是一幅精美的艺术画作，描绘了..."
    - "作品采用传统绘画技法，色彩丰富，构图精巧，展现了艺术家深厚的功底和独特的审美视角。"
    - "这件作品通过独特的艺术表现形式，展现了深厚的文化底蕴和艺术家的创作理念，为观众提供了丰富的视觉体验和审美享受。"
    """
    # 这些是已知的填充文本模式
    filling_patterns = [
        r'这是一幅精美的艺术画作，描绘了',
        r'这是一幅艺术画作，描绘了',
        r'这是一幅精美的艺术画作，',
        r'这是一幅精美的艺术作品，',
        r'这是一幅艺术作品，',
        r'作品采用传统绘画技法，色彩丰富，构图精巧，展现了艺术家深厚的功底和独特的审美视角。',
        r'这件作品通过独特的艺术表现形式，展现了深厚的文化底蕴和艺术家的创作理念，为观众提供了丰富的视觉体验和审美享受。',
        r'。作品采用传统绘画技法，色彩丰富，构图精巧，展现了艺术家深厚的功底和独特的审美视角。',
        r'。这件作品通过独特的艺术表现形式，展现了深厚的文化底蕴和艺术家的创作理念，为观众提供了丰富的视觉体验和审美享受。',
        r'，为观众提供了丰富的视觉体验和审美享受',
        r'，展现了深厚的文化底蕴和艺术家的创作理念',
        r'具有高度视觉和内容吸引力',
        r'视觉效果明显，具有强烈的视觉吸引力',
        r'，画面中呈现出深厚的文化底蕴',
        r'为观众提供了深刻的视觉体验',
        r'，通过其独特的艺术表现形式',
    ]

    cleaned = features
    for pattern in filling_patterns:
        cleaned = re.sub(pattern, '', cleaned)
        cleaned = cleaned.strip()

    # 清理多余的句号和连接词
    cleaned = re.sub(r'^[。，、]\s*', '', cleaned)
    cleaned = re.sub(r'\s*[。，、]\s*$', '', cleaned)
    cleaned = cleaned.strip()

    return cleaned


def find_original_feature(name: str, original_exhibits: Dict[str, str]) -> str:
    """
    从原始数据中查找对应的展品描述

    支持模糊匹配，因为可能存在细微差异
    """
    # 精确匹配
    if name in original_exhibits:
        return original_exhibits[name]

    # 尝试移除括号内容后匹配
    base_name = re.sub(r'\s*[【(].*?[】）]\s*', '', name)
    if base_name in original_exhibits:
        return original_exhibits[base_name]

    # 部分匹配（提取关键词）
    for orig_name, orig_features in original_exhibits.items():
        # 提取核心名称
        core_name = re.sub(r'\s*[【(].*?[】）]\s*|（.*?）', '', orig_name)
        if core_name in name or name in core_name:
            return orig_features

    # 未找到，返回空字符串
    return ""


def build_original_lookup(original_file: Path) -> Dict[str, Dict[str, str]]:
    """
    构建原始数据的展品查找表

    Returns: {
        "task_type_exhibit_name": {"features": "..."}
    }
    """
    lookup = {}

    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
        data_list = json.loads(content)  # 这是一个JSON数组

        for data in data_list:

            prompt = data.get('prompt', '')
            task_type = 'unknown'

            # 识别任务类型
            if '规划一条包含注意力等级' in prompt or '视觉扫描路径' in prompt:
                task_type = 'plan_scan_path'
            elif '下一步预测' in prompt:
                task_type = 'predict_next'
            elif '特征归因' in prompt:
                task_type = 'attribution'

            # 提取展品描述
            exhibits = extract_exhibits_from_original(prompt)

            # 构建查找键
            for exhibit_name, features in exhibits.items():
                key = f"{task_type}_{exhibit_name}"
                lookup[key] = {"features": features}

    print(f"从原始数据提取了 {len(lookup)} 条展品描述")
    return lookup


def clean_sharegpt_file(
    input_file: Path,
    output_file: Path,
    original_lookup: Dict[str, Dict[str, str]],
    stats: Dict
):
    """
    清洗单个ShareGPT文件
    """
    cleaned_count = 0
    total_features = 0
    removed_chars = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip():
                continue

            data = json.loads(line)
            original = data.copy()

            # 处理conversations中的每条消息
            for conv in data.get('conversations', []):
                if conv.get('from') != 'human':
                    continue

                # 解析human消息中的JSON
                value = conv.get('value', '')
                json_match = re.search(r'```json\n(.+?)\n```', value, re.DOTALL)

                if not json_match:
                    continue

                try:
                    json_data = json.loads(json_match.group(1))
                except:
                    continue

                task_type = json_data.get('task', '')
                exhibits = json_data.get('exhibits', [])

                # 处理每个展品的features
                for exhibit in exhibits:
                    name = exhibit.get('name', '')
                    features = exhibit.get('features', '')
                    total_features += len(features)

                    # 统计原始长度
                    original_len = len(features)

                    # 先尝试从原始数据查找
                    original_features = ""
                    for lookup_key, lookup_data in original_lookup.items():
                        if name in lookup_key or lookup_key.endswith(name) or lookup_key.startswith(f"plan_scan_path_{name}"):
                            original_features = lookup_data.get('features', '')
                            break

                    if original_features:
                        # 使用原始描述
                        exhibit['features'] = original_features
                        cleaned_count += 1
                        removed_chars += (original_len - len(original_features))
                    else:
                        # 使用清洗规则
                        cleaned = clean_filling_text(features)
                        if cleaned != features:
                            exhibit['features'] = cleaned
                            cleaned_count += 1
                            removed_chars += (original_len - len(cleaned))

                # 重建JSON
                new_json = json.dumps(json_data, ensure_ascii=False)
                new_value = value.replace(json_match.group(1), new_json)
                conv['value'] = new_value

            # 写入清洗后的数据
            f_out.write(json.dumps(original, ensure_ascii=False) + '\n')

    stats['cleaned_exhibits'] += cleaned_count
    stats['total_features'] += total_features
    stats['removed_chars'] += removed_chars


def main():
    base_dir = Path(r"D:\Python\IROS 2026_gaze\IROS_AGENT\data\processed")

    # 文件路径
    original_file = base_dir / "finetuning" / "full_jsonl.json"
    train_input = base_dir / "sharegpt_json" / "train_sharegpt.jsonl"
    train_output = base_dir / "sharegpt_json" / "train_sharegpt_cleaned.jsonl"
    val_input = base_dir / "sharegpt_json" / "val_sharegpt.jsonl"
    val_output = base_dir / "sharegpt_json" / "val_sharegpt_cleaned.jsonl"

    print("=" * 60)
    print("ShareGPT数据集清洗工具")
    print("=" * 60)

    # 步骤1: 构建原始数据查找表
    print(f"\n步骤1: 读取原始数据...")
    print(f"  输入: {original_file}")

    if not original_file.exists():
        print(f"  错误: 文件不存在")
        return

    original_lookup = build_original_lookup(original_file)
    print(f"  完成: 提取了 {len(original_lookup)} 条展品描述")

    # 步骤2: 清洗训练集
    print(f"\n步骤2: 清洗训练集...")
    stats = {'cleaned_exhibits': 0, 'total_features': 0, 'removed_chars': 0}

    if not train_input.exists():
        print(f"  跳过: 文件不存在")
    else:
        clean_sharegpt_file(train_input, train_output, original_lookup, stats)
        print(f"  完成: {train_output}")
        print(f"  清洗展品数: {stats['cleaned_exhibits']}")

    # 步骤3: 清洗验证集
    print(f"\n步骤3: 清洗验证集...")

    if not val_input.exists():
        print(f"  跳过: 文件不存在")
    else:
        clean_sharegpt_file(val_input, val_output, original_lookup, stats)
        print(f"  完成: {val_output}")

    # 统计
    print("\n" + "=" * 60)
    print("清洗统计:")
    print("=" * 60)
    print(f"  清洗展品总数: {stats['cleaned_exhibits']}")
    print(f"  移除字符数: {stats['removed_chars']:,}")
    print(f"  原始features字符数: {stats['total_features']:,}")

    if stats['total_features'] > 0:
        reduction_rate = (stats['removed_chars'] / stats['total_features']) * 100
        print(f"  压缩率: {reduction_rate:.1f}%")

    print("\n完成!")


if __name__ == "__main__":
    main()
