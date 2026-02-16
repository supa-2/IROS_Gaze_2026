#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
眼动可视化工具 - 带完整文字标注

输入格式（参照微调数据集）:
- gaze_data: [
    {'id': 1, 'name': '展品名', 'level': 'A', 'duration': 120, 'center': (x, y)},
    ...
]
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os


def create_visualization(
    image_path: str,
    gaze_data: list,
    output_path: str = None
) -> str:
    """
    创建带完整文字标注的眼动可视化

    Args:
        image_path: 展厅图片路径
        gaze_data: 眼动数据列表
        output_path: 输出路径

    Returns:
        输出文件路径
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # 尝试加载中文字体
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ]

    def load_font(size):
        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except:
                    pass
        return ImageFont.load_default(size)

    font_large = load_font(36)
    font_medium = load_font(24)
    font_small = load_font(18)

    # 注意力等级配置
    level_colors = {
        'A': (255, 50, 50),    # 红色 - 深度关注
        'B': (255, 150, 50),   # 橙色 - 中等关注
        'C': (255, 255, 50),   # 黄色 - 一般关注
        'D': (100, 200, 100),  # 绿色 - 快速浏览
        'E': (100, 100, 255),  # 蓝色 - 一瞥而过
    }

    level_desc = {
        'A': 'A级 - 深度关注 (120s)',
        'B': 'B级 - 中等关注 (60s)',
        'C': 'C级 - 一般关注 (30s)',
        'D': 'D级 - 快速浏览 (15s)',
        'E': 'E级 - 一瞥而过 (5s)',
    }

    # 绘制连接线
    for i in range(len(gaze_data) - 1):
        x1, y1 = gaze_data[i]['center']
        x2, y2 = gaze_data[i + 1]['center']
        draw.line([x1, y1, x2, y2], fill=(200, 200, 200), width=4)

    # 绘制点和标签
    for i, item in enumerate(gaze_data):
        x, y = item['center']
        level = item['level']
        color = level_colors.get(level, (128, 128, 128))

        # 绘制圆圈
        bbox = [x - 25, y - 25, x + 25, y + 25]
        draw.ellipse(bbox, outline=color, width=5)

        # 绘制序号
        num_text = str(i + 1)
        draw.text((x - 10, y - 45), num_text, fill=(255, 255, 255),
                 font=font_large, stroke_width=2, stroke_fill=(0, 0, 0))

        # 绘制展品名称
        name = item.get('name', f'展品{i+1}')
        desc = level_desc.get(level, f'{level}级')
        label_text = f"{name}\n{desc}"

        # 绘制文字背景框
        text_x = x + 35
        text_y = y - 20
        draw.text((text_x, text_y), label_text, fill=color,
                 font=font_small, stroke_width=1, stroke_fill=(255, 255, 255))

    # 绘制标题
    title = "眼动轨迹可视化 - 观看路径预测"
    draw.text((20, 15), title, fill=(255, 255, 255),
             font=font_large, stroke_width=2, stroke_fill=(0, 0, 0))

    # 绘制图例
    legend_y = h - 180
    draw.text((20, legend_y), "注意力等级图例:",
             fill=(255, 255, 255), font=font_medium,
             stroke_width=1, stroke_fill=(0, 0, 0))

    for i, (level, desc) in enumerate(level_desc.items()):
        color = level_colors[level]
        y_pos = legend_y + 35 + i * 28

        # 颜色块
        draw.rectangle([20, y_pos, 50, y_pos + 20], fill=color)

        # 文字说明
        draw.text((60, y_pos), desc, fill=(255, 255, 255),
                 font=font_small, stroke_width=1, stroke_fill=(0, 0, 0))

    # 保存
    if output_path is None:
        os.makedirs("data/outputs", exist_ok=True)
        output_path = "data/outputs/trajectory_labeled.png"

    img.save(output_path)
    return output_path


# 全局配置
LEVEL_DESC = {
    'A': 'A级 - 深度关注 (120s)',
    'B': 'B级 - 中等关注 (60s)',
    'C': 'C级 - 一般关注 (30s)',
    'D': 'D级 - 快速浏览 (15s)',
    'E': 'E级 - 一瞥而过 (5s)',
}

def main():
    """示例：创建可视化"""

    # 示例数据（按照你的微调格式）
    gaze_data = [
        {
            'id': 1,
            'name': '丁香花',
            'level': 'A',
            'duration': 120,
            'center': (250, 200)
        },
        {
            'id': 2,
            'name': '金鱼兰',
            'level': 'B',
            'duration': 60,
            'center': (500, 280)
        },
        {
            'id': 3,
            'name': '牡丹花',
            'level': 'A',
            'duration': 120,
            'center': (750, 220)
        },
        {
            'id': 4,
            'name': '说明文字',
            'level': 'C',
            'duration': 30,
            'center': (350, 450)
        },
        {
            'id': 5,
            'name': '玉兰花开',
            'level': 'E',
            'duration': 5,
            'center': (800, 400)
        },
    ]

    output = create_visualization('data/R.jpg', gaze_data)

    print("=" * 50)
    print("眼动可视化生成完成!")
    print("=" * 50)
    print(f"\n输出文件: {output}\n")

    print("观看路径预测:")
    print("-" * 40)
    for i, item in enumerate(gaze_data):
        level = item['level']
        duration = item['duration']
        name = item['name']
        x, y = item['center']

        desc = LEVEL_DESC.get(level, '')
        print(f"{i+1}. [{level}] {name}")
        print(f"   位置: ({x}, {y})")
        print(f"   预计停留: {duration}秒")
        print(f"   {desc}")
        print()


if __name__ == "__main__":
    main()
