# -*- coding: utf-8 -*-
"""
眼动热力图完整测试

使用高斯模糊生成平滑热力图叠加在原图上
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
from PIL import Image, ImageDraw, ImageFont
import urllib.request

# 导入热力图可视化器
from skills.visualization.pixel_heatmap import GazeHeatmapVisualizer


def download_test_image():
    """下载一张博物馆测试图片"""
    url = "https://images.unsplash.com/photo-1566127444979-b3d2b654e3d8?w=1200&h=800&fit=crop"
    output_path = "data/test_museum.jpg"

    os.makedirs("data", exist_ok=True)

    try:
        print("[Download] Fetching museum image...")
        urllib.request.urlretrieve(url, output_path)
        print(f"[Download] Saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"[Download] Failed: {e}")
        print("[Fallback] Creating synthetic test image...")
        return create_synthetic_image()


def create_synthetic_image():
    """创建模拟博物馆场景"""
    output_path = "data/test_museum_synthetic.jpg"

    # 创建画布 - 博物馆米色背景
    img = Image.new("RGB", (1200, 800), color="#F5F5DC")
    draw = ImageDraw.Draw(img)

    # 模拟展厅布局

    # 左侧展柜 - 青铜器
    draw.rectangle([100, 150, 350, 500], fill="#8B7355", outline="#4A3728", width=3)
    draw.rectangle([150, 200, 300, 350], fill="#CD7F32", outline="#8B6914", width=2)

    # 中间展台 - 陶瓷
    draw.rectangle([450, 200, 700, 500], fill="#DEB887", outline="#8B7355", width=3)
    draw.ellipse([500, 250, 650, 400], fill="#87CEEB", outline="#4682B4", width=2)

    # 右侧展墙 - 书画
    draw.rectangle([800, 100, 1100, 600], fill="#FFFAF0", outline="#DAA520", width=3)
    draw.rectangle([850, 150, 1050, 400], fill="#FFF8DC", outline="#BDB76B", width=2)

    # 说明牌
    draw.rectangle([400, 550, 750, 650], fill="#2F4F4F", outline="#1C3333", width=2)

    # 保存
    img.save(output_path, quality=95)
    print(f"[Created] Synthetic museum image: {output_path}")

    return output_path


def main():
    print("=" * 70)
    print("  Eye Tracking Heatmap Test")
    print("  Gaussian Blur + Overlay on Original Image")
    print("=" * 70)

    # 1. 获取测试图片
    print("\n[Step 1] Loading test image...")
    image_path = download_test_image()

    # 2. 初始化可视化器
    print("\n[Step 2] Initializing visualizer...")
    viz = GazeHeatmapVisualizer(image_path, sigma=60)
    print(f"    Image size: {viz.image_width} x {viz.image_height}")

    # 3. 添加 VLM 识别的区域
    print("\n[Step 3] Adding VLM regions...")
    vlm_regions = [
        {"id": "exhibit_a", "label": "Bronze Display", "type": "Exhibit", "bbox": [100, 150, 350, 500]},
        {"id": "exhibit_b", "label": "Pottery Stand", "type": "Exhibit", "bbox": [450, 200, 700, 500]},
        {"id": "exhibit_c", "label": "Painting Wall", "type": "Exhibit", "bbox": [800, 100, 1100, 600]},
        {"id": "label_1", "label": "Info Sign", "type": "Label", "bbox": [400, 550, 750, 650]},
    ]
    viz.add_vlm_regions(vlm_regions)
    print(f"    Added {len(vlm_regions)} regions")

    # 4. 模拟眼动数据 - 添加多个凝视点
    print("\n[Step 4] Adding gaze fixation points...")

    # 青铜器区域 - 观看最久
    viz.add_gaze_record("exhibit_a", 45.0, x=200, y=280)
    viz.add_gaze_record("exhibit_a", 30.0, x=250, y=320)
    viz.add_gaze_record("exhibit_a", 25.0, x=180, y=350)
    viz.add_gaze_record("exhibit_a", 20.0, x=220, y=300)

    # 陶瓷区域 - 中等观看
    viz.add_gaze_record("exhibit_b", 35.0, x=550, y=320)
    viz.add_gaze_record("exhibit_b", 28.0, x=580, y=350)
    viz.add_gaze_record("exhibit_b", 15.0, x=520, y=300)

    # 画墙区域 - 简单浏览
    viz.add_gaze_record("exhibit_c", 20.0, x=920, y=250)
    viz.add_gaze_record("exhibit_c", 18.0, x=950, y=300)
    viz.add_gaze_record("exhibit_c", 12.0, x=880, y=280)

    # 说明牌 - 快速扫视
    viz.add_gaze_record("label_1", 8.0, x=550, y=590)
    viz.add_gaze_record("label_1", 5.0, x=600, y=610)

    print(f"    Total fixation points: {len(viz.fixations)}")

    # 5. 获取统计信息
    print("\n[Step 5] Region statistics:")
    stats = viz.get_region_statistics()
    for region_id, stat in stats.items():
        print(f"    {stat['label']}:")
        print(f"      Fixations: {stat['fixation_count']}")
        print(f"      Total duration: {stat['total_duration']:.1f}s")
        print(f"      Avg duration: {stat['avg_duration']:.1f}s")

    # 6. 生成并排对比图
    print("\n[Step 6] Generating side-by-side comparison...")
    output_compare = viz.visualize_side_by_side(
        output_path="data/outputs/gaze_comparison.png",
        show=False
    )

    # 7. 生成单独叠加图
    print("\n[Step 7] Generating overlay visualization...")
    output_overlay = viz.visualize_overlay(
        output_path="data/outputs/gaze_overlay.png",
        show=False,
        alpha=0.6,
        cmap='jet'
    )

    print("\n" + "=" * 70)
    print("  Test Complete!")
    print(f"  Comparison: {output_compare}")
    print(f"  Overlay: {output_overlay}")
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
