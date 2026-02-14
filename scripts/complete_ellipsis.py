"""
补全省略号数据
"""
import json
import re

# 定义补全规则
ELLIPSIS_COMPLETION = {
    "耕织图": "在此展项中，通过对中国古代耕织图的数字活化，展示中国农耕文化的丰富多彩，反映出古代人民在农耕和织布中的勤劳与智慧。",

    "耕织图-多媒体动态机械装置": "在此展项中，通过对中国古代耕织图的数字活化，展示中国农耕文化的丰富多彩，反映出古代人民在农耕和织布中的勤劳与智慧。",

    "祝大年诗稿": "展现了祝大年的诗歌天赋，通过图形化展示诗歌意境，将文学与艺术完美融合，体现了诗画合一的艺术理念。",

    "祝大年": "祝大年是一位著名的画家，擅长描绘花卉和山水画，其作品色彩浓郁，构图饱满，具有独特的艺术风格。",

    "玉海棠": "一幅画着玉兰花开的画，挂在黑墙上，画面简洁优雅，玉兰花瓣洁白如玉。",

    "玉海棠（玉兰花开）": "一幅画着玉兰花开的画，挂在黑墙上，画面简洁优雅，玉兰花瓣洁白如玉。",

    "镜花一水占效": "切换了透视与幻觉的边界，运用虚实结合的手法，产生了镜花一水般虚幻的效果，体现了中国传统美学的意境。",
}


def complete_ellipsis(text: str) -> str:
    """
    补全省略号

    规则：
    1. 如果匹配到已知展品，用完整描述替换
    2. 否则，删除省略号及后面的内容
    """
    # 匹配已知展品
    for name, full_desc in ELLIPSIS_COMPLETION.items():
        pattern = rf'【{re.escape(name)}】:\s*([^\n]*?)\.\.\.+'
        if re.search(pattern, text):
            text = re.sub(pattern, f'【{name}】: {full_desc}', text, count=1)
            return text

    # 如果不是已知展品，删除省略号及后面的内容
    text = re.sub(r'\.\.[^.]*?(?=[\n\"\}])', '', text)  # 删除省略号到换行符/引号/括号
    text = re.sub(r'\.\.+$', '', text)  # 删除结尾的省略号

    return text


def process_sample(sample: dict) -> dict:
    """处理单个样本，补全省略号"""
    input_text = sample.get('input', '')
    instruction = sample.get('instruction', '')

    # 补全 input
    if '.....' in input_text:
        sample['input'] = complete_ellipsis(input_text)

    # 补全 instruction（可能包含展品名称+省略号）
    if '.....' in instruction:
        sample['instruction'] = complete_ellipsis(instruction)

    return sample


if __name__ == '__main__':
    # 读取数据
    input_path = 'data/processed/gaze_cleaned.json'
    output_path = 'data/processed/gaze_cleaned_full.json'

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f'总样本数: {len(data)}')

    # 统计原始省略号数量
    original_ellipsis_count = sum(1 for s in data if '.....' in s.get('input', ''))
    print(f'原始省略号数量: {original_ellipsis_count}')

    # 处理所有数据
    processed_data = []
    for sample in data:
        processed = process_sample(sample)
        processed_data.append(processed)

    # 验证
    final_ellipsis_count = sum(1 for s in processed_data if '.....' in s.get('input', ''))
    print(f'修复后剩余省略号: {final_ellipsis_count}')
    print(f'成功补全: {original_ellipsis_count - final_ellipsis_count}')

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f'\n已保存到: {output_path}')
