"""
使用 uv 环境处理 OS.xls 文件
"""
import xlrd
import xlwt
import os

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


def complete_ellipsis(text: str, exhibit_name: str) -> str:
    """补全省略号"""
    if '.....' not in text:
        return text

    for name, full_desc in ELLIPSIS_COMPLETION.items():
        if name in exhibit_name:
            return text.split('.....')[0].strip() + full_desc

    return text.split('.....')[0].strip()


def process_xls():
    """处理 OS.xls 文件"""
    input_file = 'skills/topology/assets/OS.xls'
    output_file = 'skills/topology/assets/OS_fixed.xls'

    print(f"[INFO] 读取文件: {input_file}")

    # 读取
    workbook = xlrd.open_workbook(input_file)
    sheet = workbook.sheet_by_index(0)

    print(f"[INFO] 总行数: {sheet.nrows}")
    print(f"[INFO] 总列数: {sheet.ncols}")

    # 获取列名
    headers = [str(sheet.cell_value(0, col)) for col in range(sheet.ncols)]
    print(f"[INFO] 列名: {headers}")

    # 查找关键列
    name_col = None
    features_col = None

    for i, h in enumerate(headers):
        if 'Name' in h or 'name' in h or '名称' in h:
            name_col = i
        if 'Visual' in h or 'visual' in h or '视觉' in h:
            features_col = i

    print(f"[INFO] Name 列: {name_col}")
    print(f"[INFO] Visual_Features 列: {features_col}")

    if features_col is None:
        print("[ERROR] 未找到 Visual_Features 列")
        return

    # 创建新工作簿
    new_wb = xlwt.Workbook(encoding='utf-8')
    new_sheet = new_wb.add_sheet('Sheet1')

    # 写入表头
    for col, h in enumerate(headers):
        new_sheet.write(0, col, h)

    # 处理数据
    ellipsis_count = 0
    completed_count = 0

    for row in range(1, sheet.nrows):
        name = sheet.cell_value(row, name_col) if name_col is not None else ""
        features = str(sheet.cell_value(row, features_col))

        if '.....' in features:
            ellipsis_count += 1
            original = features
            features = complete_ellipsis(features, str(name))

            if features != original:
                completed_count += 1
                print(f"[FIXED] 行{row}: {name}")

        # 写入所有列
        for col in range(sheet.ncols):
            if col == features_col:
                new_sheet.write(row, col, features)
            else:
                new_sheet.write(row, col, sheet.cell_value(row, col))

    # 保存
    new_wb.save(output_file)

    print(f"\n[SUCCESS] 完成!")
    print(f"[INFO] 发现省略号: {ellipsis_count}")
    print(f"[INFO] 成功补全: {completed_count}")
    print(f"[INFO] 输出: {output_file}")


if __name__ == '__main__':
    process_xls()
