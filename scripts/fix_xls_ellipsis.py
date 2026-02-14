"""
直接处理 OS.xls 文件中的省略号
使用 xlrd 直接读取和写入
"""
import sys
import os

# 尝试导入 xlrd
try:
    import xlrd
    import xlwt
except ImportError:
    print("正在安装 xlrd 和 xlwt...")
    os.system("pip install xlrd xlwt")
    import xlrd
    import xlwt

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

    Args:
        text: 包含省略号的文本
        exhibit_name: 展品名称

    Returns:
        补全后的文本
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


def process_xls_file(input_path: str, output_path: str):
    """
    处理 XLS 文件，补全省略号

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    print(f"[INFO] 正在读取文件: {input_path}")

    # 读取原始文件
    workbook = xlrd.open_workbook(input_path)
    sheet = workbook.sheet_by_index(0)

    # 获取列名
    headers = []
    for col in range(sheet.ncols):
        headers.append(sheet.cell_value(0, col))

    print(f"[INFO] 列名: {headers}")
    print(f"[INFO] 总行数: {sheet.nrows}")

    # 查找 Visual_Features 列
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

    # 创建新的工作簿
    new_workbook = xlwt.Workbook(encoding='utf-8')
    new_sheet = new_workbook.add_sheet('Sheet1')

    # 写入表头
    for col, header in enumerate(headers):
        new_sheet.write(0, col, str(header))

    # 处理数据行
    ellipsis_count = 0
    completed_count = 0

    for row in range(1, sheet.nrows):
        # 复制所有列
        for col in range(sheet.ncols):
            cell_value = sheet.cell_value(row, col)

            # 如果是 Visual_Features 列，检查是否有省略号
            if col == visual_features_col:
                exhibit_name = sheet.cell_value(row, exhibit_name_col)
                if '.....' in str(cell_value):
                    ellipsis_count += 1
                    original_value = str(cell_value)
                    cell_value = complete_ellipsis_in_text(str(cell_value), str(exhibit_name))
                    if cell_value != original_value:
                        completed_count += 1
                        print(f"[FIXED] 行 {row}: {exhibit_name}")
                        print(f"  原始: {original_value[:50]}...")
                        print(f"  补全: {cell_value[:50]}...")

            new_sheet.write(row, col, cell_value)

    # 保存新文件
    new_workbook.save(output_path)

    print(f"\n[INFO] 处理完成!")
    print(f"[INFO] 发现省略号: {ellipsis_count} 处")
    print(f"[INFO] 成功补全: {completed_count} 处")
    print(f"[INFO] 输出文件: {output_path}")


if __name__ == '__main__':
    input_file = 'skills/topology/assets/OS.xls'
    output_file = 'skills/topology/assets/OS_fixed.xls'

    print("="*80)
    print("OS.xls Ellipsis Fixer")
    print("="*80)

    process_xls_file(input_file, output_file)

    print("\n" + "="*80)
    print("完成!")
    print("="*80)
