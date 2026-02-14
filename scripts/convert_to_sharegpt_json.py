"""
转换为 ShareGPT + JSON 结构化格式

特点：
1. ShareGPT 对话格式（LLaMA-Factory 支持）
2. JSON 结构化输入输出
3. 自动提取关键词区分重复名称
4. 保留完整的 features 描述
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def extract_keyword_from_description(name: str, desc: str) -> str:
    """
    从描述中提取关键词，用于区分重复的展品名称

    规则：
    1. 说明文字：提取引号中的内容
    2. 画作类：提取主题词
    3. 其他：提取前几个关键词
    """
    if not desc or not desc.strip():
        return ""

    desc = desc.strip()

    # 规则 1: 提取引号中的内容（主要用于说明文字）
    quote_matches = re.findall(r'["«『](.*?)["」』]', desc)
    if quote_matches:
        return quote_matches[0][:10]  # 最长10字符

    # 规则 2: 提取逗号/句号前的关键词
    if '，' in desc:
        first_part = desc.split('，')[0].strip()
        if first_part and len(first_part) < 20:
            return first_part

    if ',' in desc:
        first_part = desc.split(',')[0].strip()
        if first_part and len(first_part) < 20:
            return first_part

    # 规则 3: 对于特定类型，提取特征词
    if name == '人':
        # 人：抽烟的人、盘腿坐着的人
        if '抽烟' in desc:
            return '抽烟'
        elif '盘腿' in desc:
            return '盘腿坐'
        elif '站立' in desc:
            return '站立'
        else:
            return desc[:8]

    # 规则 4: 取前10个字符作为关键词
    return desc[:10]


def process_exhibits(input_text: str) -> List[Dict]:
    """
    处理输入文本，提取展品列表

    返回格式：
    [
      {"name": "丁香花", "features": "..."},
      {"name": "说明文字（极乐鸟）", "features": "..."}
    ]
    """
    # 提取 【名称】: 描述
    pattern = r'- 【(.*?)】:\s*(.*?)(?=\n- 【|$)'
    matches = re.findall(pattern, input_text, re.DOTALL)

    exhibits = []
    name_count = defaultdict(int)  # 统计每个名称出现的次数

    # 第一遍：统计名称出现次数
    for name, desc in matches:
        desc = desc.strip()
        if desc:
            name_count[name] += 1

    # 第二遍：构建展品列表
    name_occurrences = defaultdict(int)  # 记录每个名称出现的第几次

    for name, desc in matches:
        desc = desc.strip()
        if not desc:
            continue

        name_occurrences[name] += 1
        processed_name = name

        # 如果名称重复，添加关键词区分
        if name_count[name] > 1:
            keyword = extract_keyword_from_description(name, desc)
            if keyword:
                processed_name = f"{name}（{keyword}）"

        exhibits.append({
            "name": processed_name,
            "features": desc
        })

    return exhibits


def generate_reasoning(features: str, level: str) -> str:
    """
    根据 features 和 attention_level 生成 reasoning

    Args:
        features: 视觉特征描述
        level: 注意力等级 (A/B/C/D/E)

    Returns:
        reasoning: 解释性文本
    """
    # 提取特征关键词
    has_dynamic = any(kw in features for kw in ['动态', '互动', '装置', '影像', '多媒体'])
    has_vivid_color = any(kw in features for kw in ['鲜艳', '饱满', '彩', '色彩', '红'])
    has_text = any(kw in features for kw in ['文字', '说明', '介绍', '展签', '文字说明'])
    has_detail = any(kw in features for kw in ['精细', '详细', '丰富', '复杂'])
    has_prominent = any(kw in features for kw in ['中央', '显眼', '突出', '重要'])
    is_painting = any(kw in features for kw in ['画', '绘画', '作品', '画作'])

    # 根据等级生成
    if level == 'A':
        if has_text:
            return "提供核心背景信息，需要仔细阅读理解"
        elif has_dynamic:
            return "互动性强，内容丰富，吸引深度参与"
        elif has_vivid_color and has_detail:
            return "色彩饱满且细节丰富，具有强烈视觉冲击力"
        elif has_detail:
            return "内容丰富详实，需要长时间仔细观看"
        else:
            return "核心展品，具有高度视觉和内容吸引力"

    elif level == 'B':
        if has_dynamic:
            return "动态/互动效果，视觉吸引力强"
        elif has_vivid_color:
            return "色彩鲜明，具有良好的视觉吸引力"
        elif has_text:
            return "提供辅助说明信息，值得阅读"
        elif is_painting:
            return "画作艺术性强，视觉吸引力好"
        else:
            return "视觉特征明显，吸引中等关注"

    elif level == 'C':
        if is_painting:
            return "一般性画作，浏览式观看"
        else:
            return "普通展品，一般性关注"

    elif level == 'D':
        return "次要展品，快速浏览"

    elif level == 'E':
        return "边缘展品，一瞥而过"

    return "具有中等视觉吸引力"


def parse_output_to_json(output_text: str, name_mapping: Dict[str, str] = None) -> List[Dict]:
    """
    解析输出文本为 JSON 格式

    支持多种输出格式：
    1. 热度路径规划:\n1. [B] 丁香花\n2. [A] 说明文字
    2. 下一步预测: [C] 鸟类
    3. 特征归因: ...

    Args:
        output_text: 输出文本
        name_mapping: 名称映射字典，原始名称 -> 处理后的名称
    """
    if name_mapping is None:
        name_mapping = {}

    result = []

    # 格式1: 热度路径规划
    if '热度路径规划' in output_text or '热度路径' in output_text:
        lines = output_text.split('\n')
        for line in lines:
            # 匹配: 1. [B] 丁香花
            match = re.match(r'\d+\.\s*\[([A-E])\]\s*(.+)', line.strip())
            if match:
                level, name = match.groups()
                name = name.strip()
                # 使用映射后的名称
                mapped_name = name_mapping.get(name, name)
                result.append({
                    "name": mapped_name,
                    "attention_level": level
                })

    # 格式2: 下一步预测
    elif '下一步预测' in output_text:
        match = re.search(r'下一步预测:\s*\[([A-E])\]\s*(.+)', output_text)
        if match:
            level, name = match.groups()
            name = name.strip()
            # 使用映射后的名称
            mapped_name = name_mapping.get(name, name)
            result.append({
                "name": mapped_name,
                "attention_level": level
            })

    # 格式3: 特征归因（识别并提取）
    elif '识别并提取' in output_text or '特征归因' in output_text:
        # 解析: 特征归因: 目标【丁香花】被标记为 B 级热点。其对应的视觉显著性特征包括：...。这些特征在当前场景中具有较高的视觉权重。
        match = re.search(r'目标【(.*?)】被标记为 ([A-E]) 级热点。其对应的视觉显著性特征包括：(.*?)。', output_text)
        if match:
            name, level, features = match.groups()
            name = name.strip()
            features = features.strip()

            # 使用映射后的名称
            mapped_name = name_mapping.get(name, name)

            # 生成 reasoning
            reasoning = generate_reasoning(features, level)

            result.append({
                "name": mapped_name,
                "attention_level": level,
                "visual_features": features,
                "reasoning": reasoning
            })

    return result


def convert_to_sharegpt_json(
    input_path: str,
    output_dir: str,
    train_test_split: float = 0.9
) -> Dict:
    """
    转换为 ShareGPT + JSON 结构化格式
    """
    print(f"[INFO] Loading data from: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[INFO] Total samples: {len(data)}")

    # 转换数据
    sharegpt_data = []
    error_count = 0

    for sample in data:
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        output_text = sample['output']

        # 处理输入：提取展品列表
        exhibits = process_exhibits(input_text)

        if not exhibits:
            error_count += 1
            continue

        # 创建名称映射：原始名称 -> 处理后的名称
        name_mapping = {}
        for ex in exhibits:
            # 提取原始名称（去掉关键词后缀）
            original_name = re.sub(r'（.*）', '', ex['name'])
            name_mapping[original_name] = ex['name']

        # 构造输入 JSON
        if '路径' in instruction or '扫描' in instruction:
            task_type = "plan_scan_path"
            input_json = {
                "task": task_type,
                "exhibits": exhibits
            }
        elif '预测' in instruction:
            task_type = "predict_next"

            # 提取历史行为
            history = []
            history_section = re.search(r'历史行为:(.*?)(?=$)', input_text, re.DOTALL)
            if history_section:
                history_text = history_section.group(1).strip()
                history_matches = re.findall(r'\d+\.\s*\[([A-E])\]\s*(.+)', history_text)
                for level, name in history_matches:
                    # 使用映射后的名称
                    mapped_name = name_mapping.get(name.strip(), name.strip())
                    history.append({
                        "exhibit_name": mapped_name,
                        "attention_level": level
                    })

            input_json = {
                "task": task_type,
                "exhibits": exhibits,
                "history": history
            }
        elif '识别并提取' in instruction:
            task_type = "attribution"
            input_json = {
                "task": task_type,
                "exhibits": exhibits
            }
        else:
            # 其他任务，统一格式
            input_json = {
                "task": instruction,
                "exhibits": exhibits
            }

        # 处理输出：转换为 JSON，并使用名称映射
        output_list = parse_output_to_json(output_text, name_mapping)

        # 构造输出 JSON
        if '路径' in instruction or '扫描' in instruction:
            output_json = {
                "scan_path": output_list
            }
        elif '预测' in instruction:
            if output_list:
                output_json = {
                    "prediction": output_list[0] if output_list else {}
                }
            else:
                output_json = {"prediction": {}}
        elif '识别并提取' in instruction:
            if output_list:
                output_json = {
                    "attribution": output_list[0] if output_list else {}
                }
            else:
                output_json = {"attribution": {}}
        else:
            output_json = {
                "result": output_list
            }

        # 转换为 ShareGPT 格式
        sharegpt_data.append({
            "conversations": [
                {
                    "from": "human",
                    "value": f"```json\n{json.dumps(input_json, ensure_ascii=False, indent=2)}\n```"
                },
                {
                    "from": "gpt",
                    "value": f"```json\n{json.dumps(output_json, ensure_ascii=False, indent=2)}\n```"
                }
            ]
        })

    print(f"[INFO] Successfully converted: {len(sharegpt_data)} samples")
    print(f"[INFO] Errors/Skipped: {error_count} samples")

    # 划分训练集和验证集
    split_idx = int(len(sharegpt_data) * train_test_split)
    train_data = sharegpt_data[:split_idx]
    val_data = sharegpt_data[split_idx:]

    print(f"\n[INFO] Train set: {len(train_data)} samples")
    print(f"[INFO] Validation set: {len(val_data)} samples")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练集 (JSONL)
    train_file = output_dir / "train_sharegpt.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[INFO] Saved train set to: {train_file}")

    # 保存验证集
    val_file = output_dir / "val_sharegpt.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[INFO] Saved validation set to: {val_file}")

    # 保存完整数据（JSON 格式）
    full_file = output_dir / "full_sharegpt.json"
    with open(full_file, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved full data to: {full_file}")

    # 显示示例
    print("\n[INFO] Example (first sample):")
    print("="*80)
    print(json.dumps(sharegpt_data[0], ensure_ascii=False, indent=2)[:1000])
    print("...")
    print("="*80)

    return {
        'total_samples': len(sharegpt_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'train_file': str(train_file),
        'val_file': str(val_file),
        'full_file': str(full_file)
    }


if __name__ == '__main__':
    input_path = r'd:\Python\IROS 2026_gaze\IROS_AGENT\data\processed\gaze_cleaned_full.json'
    output_dir = r'd:\Python\IROS 2026_gaze\IROS_AGENT\data\processed\sharegpt_json'

    print("="*80)
    print("Convert to ShareGPT + JSON Structured Format")
    print("="*80)

    stats = convert_to_sharegpt_json(
        input_path,
        output_dir,
        train_test_split=0.9
    )

    print("\n" + "="*80)
    print("[SUCCESS] Conversion completed!")
    print("="*80)
