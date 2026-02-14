# -*- coding: utf-8 -*-
"""
Remove all ellipsis from OS.xls
"""
import xlrd
import xlwt
import re
import shutil

# Extended completion rules
ELLIPSIS_COMPLETION = {
    "耕织图": "在此展项中，通过对中国古代耕织图的数字活化，展示中国农耕文化的丰富多彩，反映出古代人民在农耕和织布中的勤劳与智慧。",
    "祝大年诗稿": "展现了祝大年的诗歌天赋，通过图形化展示诗歌意境，将文学与艺术完美融合，体现了诗画合一的艺术理念。",
    "祝大年": "祝大年是一位著名的画家，擅长描绘花卉和山水画，其作品色彩浓郁，构图饱满，具有独特的艺术风格。",
    "玉海棠": "一幅画着玉兰花开的画，挂在黑墙上，画面简洁优雅，玉兰花瓣洁白如玉。",
    "镜花一水占效": "切换了透视与幻觉的边界，运用虚实结合的手法，产生了镜花一水般虚幻的效果，体现了中国传统美学的意境。",
    "海市蜃楼与空间效果": "通过变换的传统动态幻化展示，营造出海市蜃楼与空间交错的虚拟自然景象，展示了虚实结合的空间概念和东方美学意境。",
    "墨韵流芳": "展现了中国传统水墨画的艺术魅力，通过墨色的浓淡变化和笔触的流畅性，表现出水墨流动的韵律感和自然美。",
    "标本表现方案": "标本表现方案体现了自然真实的科学精神，通过精细的制作工艺展示自然界的多样性，强调了科学教育的重要性和审美价值。",
    "耕织图展区视觉设计": "以中国古代耕织图为参考，在绘制耕织图的画面上使用投影映射技术，展示出光影变化和空间布局的层次，增强了互动效果和传统文化氛围。",
    "景泰蓝制作": "展示景泰蓝的传统制作工艺，通过图文结合的方式详细介绍了这一非物质文化遗产的制作流程，体现了中国传统手工艺的精湛技艺和文化传承。",
    "雕塑说明": "作品表达了艺术家的构思，通过指代英雄成功的纪念性，象征着特定的艺术理念，表达了对自然与人文的敬意。",
    "祝大年与花卉数据动态投影": "结合了祝大年的花卉艺术作品与动态投影技术，创造出沉浸式的视觉体验，通过光影变化和数字技术将传统绘画与现代科技完美融合。",
    "雕刻镂空与光影的动态投影": "在可透光的介质上进行多层次雕刻，当光线穿过时产生变化的阴影，通过投影的形式营造出自然场景的沉浸式光影体验。",
}


def remove_ellipsis_completely(text):
    """Remove all ellipsis patterns"""
    # Remove various ellipsis patterns
    patterns = [r'\.{2,}$', r'\.{3,}', r'\.\.\.+']
    
    result = text
    for pattern in patterns:
        while re.search(pattern, result):
            result = re.sub(pattern, '.', result)
    
    # Clean up multiple periods
    result = re.sub(r'\.+', '.', result)
    result = result.strip('.')
    if result and not result.endswith('.'):
        result += '.'
    
    return result


def smart_complete(text, exhibit_name):
    """Smart completion using known rules"""
    for name, full_desc in ELLIPSIS_COMPLETION.items():
        if name in exhibit_name:
            text = remove_ellipsis_completely(text)
            if len(text) < 50:
                return full_desc
            return text
    
    return remove_ellipsis_completely(text)


def process_xls():
    """Process OS.xls file"""
    input_file = 'skills/topology/assets/OS.xls'
    
    print(f"[INFO] Reading: {input_file}")
    
    workbook = xlrd.open_workbook(input_file)
    sheet = workbook.sheet_by_index(0)
    
    print(f"[INFO] Total rows: {sheet.nrows}")
    
    name_col = 1
    features_col = 3
    
    new_wb = xlwt.Workbook(encoding='utf-8')
    new_sheet = new_wb.add_sheet('Sheet1')
    
    # Copy header
    for col in range(sheet.ncols):
        new_sheet.write(0, col, sheet.cell_value(0, col))
    
    ellipsis_count = 0
    fixed_count = 0
    
    for row in range(1, sheet.nrows):
        name = str(sheet.cell_value(row, name_col))
        features = str(sheet.cell_value(row, features_col))
        
        has_ellipsis = '..' in features
        
        if has_ellipsis:
            ellipsis_count += 1
            original = features
            features = smart_complete(features, name)
            
            if features != original:
                fixed_count += 1
                print(f"\n[FIXED] Row {row}: {name}")
                print(f"  Before: {original[:60]}...")
                print(f"  After:  {features[:60]}...")
        
        # Write all columns
        for col in range(sheet.ncols):
            if col == features_col:
                new_sheet.write(row, col, features)
            else:
                new_sheet.write(row, col, sheet.cell_value(row, col))
    
    # Backup
    backup_file = input_file.replace('.xls', '_before_ellipsis_removal.xls')
    shutil.copy(input_file, backup_file)
    print(f"\n[INFO] Backed up to: {backup_file}")
    
    # Save
    new_wb.save(input_file)
    
    print(f"\n[SUCCESS] Done!")
    print(f"[INFO] Found rows with ellipsis: {ellipsis_count}")
    print(f"[INFO] Fixed: {fixed_count}")
    
    # Verify
    wb_verify = xlrd.open_workbook(input_file)
    sheet_verify = wb_verify.sheet_by_index(0)
    
    remaining = 0
    for row in range(1, sheet_verify.nrows):
        features = str(sheet_verify.cell_value(row, 3))
        if '..' in features:
            remaining += 1
    
    if remaining == 0:
        print(f"[VERIFY] No ellipsis found!")
    else:
        print(f"[WARNING] Still {remaining} rows with ellipsis")


if __name__ == '__main__':
    process_xls()
