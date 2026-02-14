# -*- coding: utf-8 -*-
"""
真实博物馆场景 + 眼动热力图测试
"""

import sys
import os
import io

# UTF-8 output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# 导入热力图可视化器
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from skills.visualization.pixel_heatmap import GazeHeatmapVisualizer


def create_realistic_museum_scene():
    """创建逼真的博物馆场景"""
    output_path = "data/realistic_museum.jpg"
    os.makedirs("data", exist_ok=True)

    # 创建画布
    width, height = 1600, 1000
    img = Image.new("RGB", (width, height), color="#F5F5DC")
    draw = ImageDraw.Draw(img)

    # 墙面渐变背景
    for i in range(700):
        g = 245 - int(i * 0.015)
        draw.rectangle([(0, i), (width, i+1)], fill=f"#{g:02x}{g:02x}{g-5:02x}")

    # 地板
    draw.rectangle([0, 700, width, height], fill="#8B7355")
    draw.rectangle([0, 700, width, 705], fill="#A0522D")

    # 左侧展柜 - 青铜器
    draw.rectangle([100, 150, 450, 650], fill="#4A3728", outline="#2F1F1F", width=5)
    draw.rectangle([115, 165, 435, 550], fill="#E6E6FA", outline="#B0B0FF", width=2)
    # 青铜展品
    draw.ellipse([200, 250, 350, 400], fill="#8B4513", outline="#CD7F32", width=8)
    draw.ellipse([230, 280, 320, 370], fill="#A0522D", outline="#CD7F32", width=3)

    # 中间展台 - 陶瓷
    draw.rectangle([550, 300, 900, 650], fill="#DEB887", outline="#8B7355", width=5)
    # 陶瓷花瓶
    draw.ellipse([650, 320, 800, 450], fill="#87CEEB", outline="#4682B4", width=6)
    draw.rectangle([680, 280, 770, 330], fill="#B0E0E6", outline="#4682B4", width=3)
    draw.ellipse([690, 300, 760, 320], fill="#87CEEB", outline="#4682B4", width=2)

    # 右侧展墙 - 书画
    draw.rectangle([1000, 100, 1450, 650], fill="#FFFAF0", outline="#DAA520", width=5)
    draw.rectangle([1100, 150, 1400, 400], fill="#FFF8DC", outline="#BDB76B", width=8)
    draw.rectangle([1120, 170, 1380, 380], fill="#FFEFD5", outline="#EEE8AA", width=3)
    # 书画内容（抽象）
    draw.ellipse([1200, 200, 1300, 280], fill="#DDA0DD", outline="#9370DB", width=4)
    draw.ellipse([1220, 250, 1280, 320], fill="#FFB6C1", outline="#DB7093", width=3)

    # 说明牌
    draw.rectangle([400, 750, 1100, 850], fill="#2F4F4F", outline="#1C3333", width=4)
    # 模拟文字行
    for x in range(450, 1000, 120):
        draw.rectangle([x, 775, x+80, 790], fill="#696969")
        draw.rectangle([x, 795, x+60, 805], fill="#696969")
        draw.rectangle([x, 815, x+70, 825], fill="#696969")

    img.save(output_path, quality=95)
    print(f"[Created] {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("  真实博物馆场景 + 眼动热力图测试")
    print("=" * 70)

    # 1. 创建/下载图片
    print("\n[Step 1] Loading museum image...")
    image_path = create_realistic_museum_scene()

    # 2. 初始化可视化器
    print("\n[Step 2] Initializing heatmap visualizer...")
    viz = GazeHeatmapVisualizer(image_path, sigma=80)
    print(f"    Image size: {viz.image_width} x {viz.image_height}")

    # 3. VLM 识别的区域
    print("\n[Step 3] Adding VLM detected regions...")
    vlm_regions = [
        {"id": "bronze", "label": "Bronze Ding", "type": "Exhibit", "bbox": [100, 150, 450, 650]},
        {"id": "pottery", "label": "Pottery Vase", "type": "Exhibit", "bbox": [550, 300, 900, 650]},
        {"id": "painting", "label": "Art Painting", "type": "Exhibit", "bbox": [1000, 100, 1450, 650]},
        {"id": "sign", "label": "Info Sign", "type": "Label", "bbox": [400, 750, 1100, 850]},
    ]
    viz.add_vlm_regions(vlm_regions)

    # 4. 眼动数据 - 分散的凝视点
    print("\n[Step 4] Adding gaze fixation points...")

    # 青铜展品 - 观看最久
    viz.add_gaze_record("bronze", 35.0, x=275, y=325)
    viz.add_gaze_record("bronze", 28.0, x=250, y=350)
    viz.add_gaze_record("bronze", 25.0, x=300, y=380)
    viz.add_gaze_record("bronze", 22.0, x=275, y=400)
    viz.add_gaze_record("bronze", 18.0, x=260, y=360)
    viz.add_gaze_record("bronze", 15.0, x=290, y=340)

    # 陶瓷展台
    viz.add_gaze_record("pottery", 30.0, x=725, y=380)
    viz.add_gaze_record("pottery", 25.0, x=750, y=400)
    viz.add_gaze_record("pottery", 20.0, x=700, y=420)
    viz.add_gaze_record("pottery", 15.0, x=730, y=360)
    viz.add_gaze_record("pottery", 12.0, x=715, y=390)

    # 书画墙 - 快速浏览
    viz.add_gaze_record("painting", 18.0, x=1250, y=275)
    viz.add_gaze_record("painting", 15.0, x=1280, y=300)
    viz.add_gaze_record("painting", 12.0, x=1220, y=320)
    viz.add_gaze_record("painting", 10.0, x=1260, y=280)
    viz.add_gaze_record("painting", 8.0, x=1240, y=290)

    # 说明牌
    viz.add_gaze_record("sign", 8.0, x=750, y=800)
    viz.add_gaze_record("sign", 6.0, x=800, y=820)
    viz.add_gaze_record("sign", 5.0, x=700, y=810)
    viz.add_gaze_record("sign", 4.0, x=780, y=805)

    print(f"    Total fixations: {len(viz.fixations)}")

    # 5. 统计信息
    print("\n[Step 5] Region statistics:")
    stats = viz.get_region_statistics()
    for region_id, stat in stats.items():
        print(f"    {stat['label']}:")
        print(f"      Fixations: {stat['fixation_count']}, Total: {stat['total_duration']:.1f}s")

    # 6. 生成对比图
    print("\n[Step 6] Generating comparison...")
    os.makedirs("data/outputs", exist_ok=True)
    viz.visualize_side_by_side(
        output_path="data/outputs/real_museum_comparison.png",
        show=False
    )

    # 7. 叠加图 (jet)
    print("\n[Step 7] Generating overlay (jet)...")
    viz.visualize_overlay(
        output_path="data/outputs/real_museum_overlay.png",
        show=False,
        alpha=0.55,
        cmap='jet'
    )

    # 8. 叠加图 (hot - 更红)
    print("\n[Step 8] Generating overlay (hot)...")
    viz.visualize_overlay(
        output_path="data/outputs/real_museum_overlay_hot.png",
        show=False,
        alpha=0.5,
        cmap='hot'
    )

    print("\n" + "=" * 70)
    print("  Test Complete!")
    print("  Outputs:")
    print("    - data/realistic_museum.jpg (original)")
    print("    - data/outputs/real_museum_comparison.png")
    print("    - data/outputs/real_museum_overlay.png")
    print("    - data/outputs/real_museum_overlay_hot.png")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
