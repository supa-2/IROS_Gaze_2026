# -*- coding: utf-8 -*-
"""
处理用户上传的真实博物馆照片 + 眼动热力图
"""

import sys
import os
import io

# UTF-8 output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# 导入热力图可视化器
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from skills.visualization.pixel_heatmap import GazeHeatmapVisualizer


def main():
    print("=" * 70)
    print("  真实博物馆照片 + 眼动热力图")
    print("=" * 70)

    # 用户上传的图片路径
    image_path = "data/R.jpg"

    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return False

    # 1. 初始化可视化器
    print("\n[Step 1] Loading image...")
    img = Image.open(image_path)
    width, height = img.size
    print(f"    Image size: {width} x {height}")

    viz = GazeHeatmapVisualizer(image_path, sigma=60)

    # 2. VLM 识别的区域（基于图片分析）
    print("\n[Step 2] Adding VLM detected regions...")

    vlm_regions = [
        # 左侧墙面画作
        {"id": "painting_1", "label": "Left Wall Painting 1", "type": "Exhibit", "bbox": [25, 355, 222, 730]},
        {"id": "painting_2", "label": "Left Wall Painting 2", "type": "Exhibit", "bbox": [235, 375, 315, 670]},
        {"id": "painting_3", "label": "Left Wall Painting 3", "type": "Exhibit", "bbox": [332, 420, 372, 605]},

        # 中间背景墙面画作
        {"id": "painting_4", "label": "Center Wall Painting", "type": "Exhibit", "bbox": [427, 450, 518, 535]},

        # 右侧墙面大幅画
        {"id": "painting_5", "label": "Right Wall Painting", "type": "Exhibit", "bbox": [698, 345, 918, 715]},

        # 雕塑与底座
        {"id": "sculpture", "label": "Center Sculpture", "type": "Exhibit", "bbox": [495, 560, 545, 835]},

        # 休息区
        {"id": "sofa", "label": "Rest Sofa", "type": "Facility", "bbox": [427, 570, 497, 665]},

        # 标识
        {"id": "sign_1", "label": "MÉBOSPACE Sign", "type": "Label", "bbox": [752, 0, 918, 180]},
        {"id": "sign_2", "label": "BC Code Sign", "type": "Label", "bbox": [502, 360, 530, 405]},
    ]

    viz.add_vlm_regions(vlm_regions)
    print(f"    Added {len(vlm_regions)} regions")

    # 3. 模拟真实眼动数据
    print("\n[Step 3] Adding simulated gaze data...")

    # 右侧大幅画 - 观众停留最久（最大视觉焦点）
    viz.add_gaze_record("painting_5", 40.0, x=808, y=530)
    viz.add_gaze_record("painting_5", 35.0, x=750, y=500)
    viz.add_gaze_record("painting_5", 32.0, x=850, y=550)
    viz.add_gaze_record("painting_5", 28.0, x=780, y=480)
    viz.add_gaze_record("painting_5", 25.0, x=820, y=580)
    viz.add_gaze_record("painting_5", 22.0, x=770, y=520)

    # 左侧第一幅画 - 详细观看
    viz.add_gaze_record("painting_1", 35.0, x=123, y=540)
    viz.add_gaze_record("painting_1", 30.0, x=150, y=500)
    viz.add_gaze_record("painting_1", 28.0, x=100, y=580)
    viz.add_gaze_record("painting_1", 25.0, x=130, y=520)
    viz.add_gaze_record("painting_1", 20.0, x=160, y=560)

    # 中间雕塑 - 中等关注
    viz.add_gaze_record("sculpture", 30.0, x=520, y=700)
    viz.add_gaze_record("sculpture", 25.0, x=500, y=750)
    viz.add_gaze_record("sculpture", 22.0, x=530, y=680)
    viz.add_gaze_record("sculpture", 18.0, x=515, y=720)
    viz.add_gaze_record("sculpture", 15.0, x=525, y=710)

    # 左侧第二幅画
    viz.add_gaze_record("painting_2", 25.0, x=275, y=520)
    viz.add_gaze_record("painting_2", 22.0, x=290, y=500)
    viz.add_gaze_record("painting_2", 18.0, x=260, y=540)

    # 休息区沙发 - 休息时的扫视
    viz.add_gaze_record("sofa", 15.0, x=462, y=617)
    viz.add_gaze_record("sofa", 12.0, x=480, y=600)
    viz.add_gaze_record("sofa", 10.0, x=445, y=630)

    # 左侧小幅画
    viz.add_gaze_record("painting_3", 18.0, x=357, y=512)
    viz.add_gaze_record("painting_3", 15.0, x=340, y=500)

    # 中间背景画
    viz.add_gaze_record("painting_4", 15.0, x=472, y=492)
    viz.add_gaze_record("painting_4", 12.0, x=490, y=480)

    # 标识牌 - 快速扫视
    viz.add_gaze_record("sign_1", 8.0, x=835, y=90)
    viz.add_gaze_record("sign_1", 5.0, x=800, y=120)
    viz.add_gaze_record("sign_2", 6.0, x=516, y=382)

    print(f"    Total fixations: {len(viz.fixations)}")

    # 4. 统计信息
    print("\n[Step 4] Region statistics:")
    stats = viz.get_region_statistics()
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_duration'], reverse=True)

    for region_id, stat in sorted_stats:
        print(f"    {stat['label']}:")
        print(f"      Fixations: {stat['fixation_count']}, Total: {stat['total_duration']:.1f}s")

    # 5. 生成输出
    print("\n[Step 5] Generating visualizations...")
    os.makedirs("data/outputs", exist_ok=True)

    # 三联对比图
    print("  - Creating comparison...")
    viz.visualize_side_by_side(
        output_path="data/outputs/R_comparison.png",
        show=False
    )

    # 叠加图 - jet 配色
    print("  - Creating overlay (jet)...")
    viz.visualize_overlay(
        output_path="data/outputs/R_overlay_jet.png",
        show=False,
        alpha=0.55,
        cmap='jet'
    )

    # 叠加图 - hot 配色（更红）
    print("  - Creating overlay (hot)...")
    viz.visualize_overlay(
        output_path="data/outputs/R_overlay_hot.png",
        show=False,
        alpha=0.5,
        cmap='hot'
    )

    # 叠加图 - viridis 配色（色盲友好）
    print("  - Creating overlay (viridis)...")
    viz.visualize_overlay(
        output_path="data/outputs/R_overlay_viridis.png",
        show=False,
        alpha=0.55,
        cmap='viridis'
    )

    print("\n" + "=" * 70)
    print("  Processing Complete!")
    print("  Outputs:")
    print("    - data/outputs/R_comparison.png")
    print("    - data/outputs/R_overlay_jet.png")
    print("    - data/outputs/R_overlay_hot.png")
    print("    - data/outputs/R_overlay_viridis.png")
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
