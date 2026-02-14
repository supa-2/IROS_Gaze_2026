# -*- coding: utf-8 -*-
"""
像素级热力图完整测试
VLM 区域识别 + 眼动数据 → 热力图叠加
"""

import sys
import os
import io

# 设置 UTF-8 输出编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# 导入我们的像素热力图可视化器
sys.path.insert(0, os.path.dirname(__file__))
from skills.visualization.pixel_heatmap import PixelHeatmapVisualizer

def main():
    print("=" * 70)
    print("  像素级热力图完整测试")
    print("  VLM 区域 + 眼动数据 → 热力图")
    print("=" * 70)

    # 1. 创建测试图片
    print("\n[1] 创建测试图片...")
    os.makedirs("data/outputs", exist_ok=True)

    test_img = Image.new("RGB", (800, 600), color="#f5f5f5")
    draw = ImageDraw.Draw(test_img)

    # 画一些简单的模拟展品
    draw.rectangle([100, 150, 300, 350], fill="#8B4513", outline="#000000")  # 青铜器
    draw.rectangle([400, 100, 550, 300], fill="#DEB887", outline="#000000")  # 陶瓷
    draw.rectangle([500, 350, 650, 450], fill="#F5F5DC", outline="#000000")  # 说明牌

    test_img_path = "data/test_museum.jpg"
    test_img.save(test_img_path)
    print(f"    测试图片已保存: {test_img_path}")

    # 2. 初始化可视化器
    print("\n[2] 初始化像素热力图可视化器...")
    viz = PixelHeatmapVisualizer(test_img_path)
    print(f"    图片尺寸: {viz.image_width} x {viz.image_height}")

    # 3. 添加 VLM 识别的区域
    print("\n[3] 添加 VLM 识别的区域...")
    vlm_regions = [
        {"id": "zone_a", "label": "青铜器区", "type": "Exhibit", "bbox": [100, 150, 300, 350]},
        {"id": "zone_b", "label": "陶瓷展区", "type": "Exhibit", "bbox": [400, 100, 550, 300]},
        {"id": "zone_c", "label": "说明牌区", "type": "Label", "bbox": [500, 350, 650, 450]},
    ]
    viz.add_vlm_regions(vlm_regions)
    print(f"    已添加 {len(vlm_regions)} 个区域")

    # 4. 添加眼动数据
    print("\n[4] 添加眼动数据...")
    gaze_records = [
        ("zone_a", 45.0),  # 青铜器区看了45秒
        ("zone_a", 30.0),  # 又看了30秒
        ("zone_b", 60.0),  # 陶瓷区看了60秒
        ("zone_c", 20.0),  # 说明牌看了20秒
    ]

    for zone_id, duration in gaze_records:
        viz.add_gaze_record(region_id=zone_id, duration=duration)

    print(f"    已添加 {len(gaze_records)} 条眼动记录")

    # 5. 获取统计信息
    print("\n[5] 区域统计:")
    stats = viz.get_region_statistics()
    for region_id, stat in stats.items():
        print(f"    {stat['label']}:")
        print(f"      观看次数: {stat['view_count']}")
        print(f"      总时长: {stat['total_duration']:.1f}s")
        print(f"      平均时长: {stat['avg_duration']:.1f}s")

    # 6. 计算热力图
    print("\n[6] 计算像素级热力图...")
    heatmap = viz.calculate_heatmap(decay_factor=0.95)
    print(f"    热力图尺寸: {heatmap.shape}")
    print(f"    热力值范围: {heatmap.min():.3f} ~ {heatmap.max():.3f}")

    # 7. 生成叠加可视化
    print("\n[7] 生成热力图叠加可视化...")
    output_path = viz.visualize_overlay(
        output_path="data/outputs/pixel_heatmap_final.png",
        show=False,
        alpha=0.5
    )
    print(f"    热力图已保存: {output_path}")

    print("\n" + "=" * 70)
    print("  测试完成!")
    print("=" * 70)

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
