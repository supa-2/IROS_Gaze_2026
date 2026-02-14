"""
独立脚本：处理 OS.xls 文件中的省略号
不依赖项目环境，使用独立的包管理
"""
import sys
import os

# 添加本地包路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

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


def complete_ellipsis_in_text(text: str, exhibit_name: str) -> str:
    """
    补全省略号
    """
    if '.....' not in text:
        return text

    # 检查是否是已知的展品
    for name, full_desc in ELLIPSIS_COMPLETION.items():
        if name in exhibit_name:
            # 删除省略号部分，用完整描述替换
            text = text.split('.....')[0].strip()
            return full_desc

    # 如果不是已知展品，删除省略号及后面的内容
    text = text.split('.....')[0].strip()
    return text


def process_csv_file(input_path: str, output_path: str):
    """
    处理 CSV 文件（如果 OS.xls 可以转换为 CSV）

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    print(f"[INFO] 正在读取文件: {input_path}")

    import csv

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("[ERROR] 文件为空")
        return

    print(f"[INFO] 列名: {rows[0]}")
    print(f"[INFO] 总行数: {len(rows)}")

    headers = rows[0]

    # 查找 Visual_Features 列和 Name 列
    visual_features_col = None
    exhibit_name_col = None

    for i, header in enumerate(headers):
        if 'Visual' in str(header) or 'visual' in str(header):
            visual_features_col = i
        if 'Name' in str(header) or 'name' in str(header):
            exhibit_name_col = i

    if visual_features_col is None:
        print("[ERROR] 未找到 Visual_Features 列")
        return

    if exhibit_name_col is None:
        print("[WARNING] 未找到 Name 列，将使用第一列")
        exhibit_name_col = 0

    print(f"[INFO] Visual_Features 列索引: {visual_features_col}")
    print(f"[INFO] Name 列索引: {exhibit_name_col}")

    # 处理数据行
    ellipsis_count = 0
    completed_count = 0

    for row in range(1, len(rows)):
        if visual_features_col < len(rows[row]):
            cell_value = str(rows[row][visual_features_col])
            exhibit_name = str(rows[row][exhibit_name_col]) if exhibit_name_col < len(rows[row]) else ""

            if '.....' in cell_value:
                ellipsis_count += 1
                original_value = cell_value
                cell_value = complete_ellipsis_in_text(cell_value, exhibit_name)
                if cell_value != original_value:
                    completed_count += 1
                    print(f"[FIXED] 行 {row}: {exhibit_name}")
                    print(f"  原始: {original_value[:50]}...")
                    print(f"  补全: {cell_value[:50]}...")

                rows[row][visual_features_col] = cell_value

    # 保存文件
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n[INFO] 处理完成!")
    print(f"[INFO] 发现省略号: {ellipsis_count} 处")
    print(f"[INFO] 成功补全: {completed_count} 处")
    print(f"[INFO] 输出文件: {output_path}")


if __name__ == '__main__':
    # 尝试使用 CSV 格式
    input_file = 'skills/topology/assets/OS.csv'
    output_file = 'skills/topology/assets/OS_fixed.csv'

    print("="*80)
    print("OS Ellipsis Fixer (CSV format)")
    print("="*80)

    # 检查 CSV 文件是否存在
    if not os.path.exists(input_file):
        print(f"[INFO] CSV 文件不存在，尝试查找...")
        # 列出 assets 目录的文件
        assets_dir = 'skills/topology/assets'
        if os.path.exists(assets_dir):
            files = os.listdir(assets_dir)
            print(f"[INFO] Assets 目录文件:")
            for f in files:
                print(f"  - {f}")

            # 检查是否有 CSV 文件
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                input_file = os.path.join(assets_dir, csv_files[0])
                output_file = input_file.replace('.csv', '_fixed.csv')
                print(f"[INFO] 找到 CSV 文件: {input_file}")
            else:
                print("[WARNING] 未找到 CSV 文件")
                print("[INFO] 提示: OS.xls 需要使用 xlrd 库读取")
                print("[INFO] 建议: 手动将 OS.xls 转换为 CSV 格式")
                print("[INFO] 或者使用已经处理好的数据: data/processed/gaze_cleaned_full.json")
        else:
            print(f"[ERROR] Assets 目录不存在: {assets_dir}")
    else:
        process_csv_file(input_file, output_file)

    print("\n" + "="*80)
    print("完成!")
    print("="*80)
