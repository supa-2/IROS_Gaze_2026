# -*- coding: utf-8 -*-
"""修正坐标后的真实照片热力图"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from skills.visualization.pixel_heatmap import GazeHeatmapVisualizer

def main():
    print("=" * 60)
    print("  修正坐标 - 真实照片热力图")
    print("=" * 60)

    image_path = "data/R.jpg"
    viz = GazeHeatmapVisualizer(image_path, sigma=50)

    # 修正后的区域坐标
    vlm_regions = [
        {"id": "left_1", "label": "Left Painting 1", "type": "Exhibit", "bbox": [50, 200, 250, 480]},
        {"id": "left_2", "label": "Left Painting 2", "type": "Exhibit", "bbox": [260, 220, 340, 460]},
        {"id": "left_3", "label": "Left Painting 3", "type": "Exhibit", "bbox": [350, 250, 420, 420]},
        {"id": "right_1", "label": "Right Large Painting", "type": "Exhibit", "bbox": [700, 180, 950, 500]},
        {"id": "center_sculpture", "label": "Center Sculpture", "type": "Exhibit", "bbox": [480, 400, 560, 550]},
        {"id": "sofa", "label": "Rest Sofa", "type": "Facility", "bbox": [420, 480, 520, 560]},
    ]
    viz.add_vlm_regions(vlm_regions)
    print(f"Added {len(vlm_regions)} regions")

    # 修正后的凝视点坐标
    print("Adding corrected gaze points...")

    # 右侧大幅画 - 最受欢迎
    viz.add_gaze_record("right_1", 45.0, x=825, y=340)
    viz.add_gaze_record("right_1", 38.0, x=780, y=300)
    viz.add_gaze_record("right_1", 35.0, x=870, y=380)
    viz.add_gaze_record("right_1", 32.0, x=850, y=280)
    viz.add_gaze_record("right_1", 28.0, x=750, y=350)
    viz.add_gaze_record("right_1", 25.0, x=800, y=400)
    viz.add_gaze_record("right_1", 22.0, x=880, y=320)

    # 左侧第一幅画 - 详细观看
    viz.add_gaze_record("left_1", 40.0, x=150, y=340)
    viz.add_gaze_record("left_1", 35.0, x=120, y=300)
    viz.add_gaze_record("left_1", 32.0, x=180, y=380)
    viz.add_gaze_record("left_1", 28.0, x=200, y=280)
    viz.add_gaze_record("left_1", 25.0, x=100, y=360)
    viz.add_gaze_record("left_1", 22.0, x=160, y=420)

    # 中间雕塑
    viz.add_gaze_record("center_sculpture", 35.0, x=520, y=475)
    viz.add_gaze_record("center_sculpture", 30.0, x=500, y=460)
    viz.add_gaze_record("center_sculpture", 28.0, x=540, y=490)
    viz.add_gaze_record("center_sculpture", 25.0, x=520, y=450)
    viz.add_gaze_record("center_sculpture", 22.0, x=510, y=510)

    # 左侧第二幅画
    viz.add_gaze_record("left_2", 30.0, x=300, y=340)
    viz.add_gaze_record("left_2", 28.0, x=280, y=310)
    viz.add_gaze_record("left_2", 25.0, x=320, y=370)
    viz.add_gaze_record("left_2", 22.0, x=290, y=380)

    # 休息沙发
    viz.add_gaze_record("sofa", 20.0, x=470, y=520)
    viz.add_gaze_record("sofa", 18.0, x=450, y=500)
    viz.add_gaze_record("sofa", 15.0, x=490, y=540)

    # 左侧第三幅画
    viz.add_gaze_record("left_3", 25.0, x=385, y=335)
    viz.add_gaze_record("left_3", 22.0, x=365, y=310)

    print(f"Total fixations: {len(viz.fixations)}")

    # 统计
    print("\nRegion statistics:")
    stats = viz.get_region_statistics()
    for rid, stat in sorted(stats.items(), key=lambda x: x[1]['total_duration'], reverse=True):
        print(f"  {stat['label']}: {stat['fixation_count']} fixations, {stat['total_duration']:.1f}s")

    # 生成
    os.makedirs("data/outputs", exist_ok=True)
    viz.visualize_side_by_side(output_path="data/outputs/R_comparison_fixed.png", show=False)
    viz.visualize_overlay(output_path="data/outputs/R_overlay_fixed.png", show=False, alpha=0.6, cmap='jet')

    # 更精确的版本（sigma=30）
    viz2 = GazeHeatmapVisualizer(image_path, sigma=30)
    viz2.add_vlm_regions(vlm_regions)
    for f in viz.fixations:
        viz2.fixations.append(f)
    viz2.visualize_overlay(output_path="data/outputs/R_overlay_precise.png", show=False, alpha=0.6, cmap='hot')

    print("\n" + "=" * 60)
    print("Fixed outputs:")
    print("  - R_comparison_fixed.png")
    print("  - R_overlay_fixed.png")
    print("  - R_overlay_precise.png")
    print("=" * 60)

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
