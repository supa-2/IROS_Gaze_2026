#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试使用 SAM2 分割结果生成热力图和眼动轨迹图
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from skills.visualization.heatmap import HeatmapVisualizer
from skills.visualization.trajectory import TrajectoryVisualizer


def load_segmentation_results(output_dir):
    """加载 SAM2 分割结果"""
    info_path = os.path.join(output_dir, "segmentation_info.json")
    centers_path = os.path.join(output_dir, "centers.json")

    if not os.path.exists(info_path):
        print(f"[!] Info file not found: {info_path}")
        return None

    with open(info_path, 'r') as f:
        info = json.load(f)

    centers = []
    if os.path.exists(centers_path):
        with open(centers_path, 'r') as f:
            centers_data = json.load(f)
            centers = centers_data.get('centers', [])

    return info, centers


def create_mock_gaze_data(centers, num_points=50):
    """基于分割中心点创建模拟眼动数据"""
    if not centers:
        return []

    gaze_data = []
    for i in range(num_points):
        # 随机选择一个中心点
        center_idx = i % len(centers)
        cx, cy = centers[center_idx]

        # 添加一些随机偏移
        offset_x = np.random.randint(-30, 31)
        offset_y = np.random.randint(-30, 31)

        # 随机停留时间
        duration = np.random.randint(5, 60)

        gaze_data.append({
            "x": cx + offset_x,
            "y": cy + offset_y,
            "duration": duration,
            "timestamp": i * 100,
            "exhibit_id": center_idx
        })

    return gaze_data


def plot_heatmap_on_image(image_path, gaze_data, output_path):
    """在原图上绘制热力图"""
    from scipy.ndimage import gaussian_filter

    # 加载原图
    img = Image.open(image_path)
    width, height = img.size

    # 创建热力图
    heatmap = np.zeros((height, width), dtype=np.float32)

    for gaze in gaze_data:
        x, y = int(gaze['x']), int(gaze['y'])
        if 0 <= x < width and 0 <= y < height:
            # 根据停留时间加权
            weight = gaze.get('duration', 1) / 60.0  # 归一化
            heatmap[y, x] += weight

    # 高斯模糊
    heatmap = gaussian_filter(heatmap, sigma=25)

    # 归一化
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # 转换为颜色图
    import matplotlib.cm as cm
    colormap = cm.get_cmap('jet')

    # 创建彩色热力图
    heatmap_img = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(3):
        heatmap_img[:, :, i] = (colormap(heatmap)[:, :, i] * 255).astype(np.uint8)
    heatmap_img[:, :, 3] = (colormap(heatmap)[:, :, 3] * 180).astype(np.uint8)  # Alpha

    # 叠加到原图
    heatmap_pil = Image.fromarray(heatmap_img, 'RGBA')
    img_rgba = img.convert('RGBA')
    combined = Image.alpha_composite(img_rgba, heatmap_pil)

    combined.convert('RGB').save(output_path)
    print(f"  [*] Heatmap saved: {output_path}")


def plot_trajectory_on_image(image_path, gaze_data, output_path):
    """在原图上绘制眼动轨迹"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 绘制轨迹线
    if len(gaze_data) > 1:
        points = [(g['x'], g['y']) for g in gaze_data]
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            # 渐变色效果，按时间顺序
            intensity = int(255 * (1 - i / len(points)))
            color = (intensity, 0, 255 - intensity)
            draw.line([x1, y1, x2, y2], fill=color, width=2)

    # 绘制停留点（大小与停留时间成正比）
    for i, gaze in enumerate(gaze_data):
        x, y = gaze['x'], gaze['y']
        duration = gaze.get('duration', 10)
        radius = max(3, min(15, duration // 5))

        # 根据时间顺序颜色渐变
        intensity = int(255 * (1 - i / len(gaze_data)))
        color = (intensity, 0, 255 - intensity)

        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color, outline=(255, 255, 255), width=2)

        # 标注序号
        if i % 5 == 0:  # 每5个点标注一次
            draw.text((x + radius + 2, y - radius), str(i), fill=(255, 0, 0))

    img.save(output_path)
    print(f"  [*] Trajectory saved: {output_path}")


def plot_detection_boxes(image_path, detections, output_path):
    """绘制检测框"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0)
    ]

    for det in detections:
        idx = det.get('id', 0)
        bbox = det.get('bbox')
        center = det.get('center')
        confidence = det.get('confidence', 0)

        if bbox:
            x1, y1, x2, y2 = bbox
            color = colors[idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 标签
            label = f"#{idx+1} ({confidence:.2f})"
            draw.text((x1, y1 - 15), label, fill=color)

        if center:
            cx, cy = center
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 0, 0))

    img.save(output_path)
    print(f"  [*] Detection boxes saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test segmentation visualization")
    parser.add_argument("--image", type=str, default="data/R.jpg", help="Original image path")
    parser.add_argument("--seg_dir", type=str, default="data/outputs/sam2_small",
                       help="Segmentation output directory")
    parser.add_argument("--output", type=str, default="data/outputs/sam2_viz",
                       help="Visualization output directory")

    args = parser.parse_args()

    print("=" * 60)
    print("SAM2 分割结果可视化测试")
    print("=" * 60)

    # 加载分割结果
    print(f"\n[*] Loading segmentation results from: {args.seg_dir}")
    result = load_segmentation_results(args.seg_dir)

    if result is None:
        print("[!] Failed to load segmentation results")
        return

    info, centers = result
    detections = info.get('detections', [])
    print(f"    Found {len(detections)} detections")
    print(f"    Found {len(centers)} center points")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 绘制检测框
    print(f"\n[*] Drawing detection boxes...")
    plot_detection_boxes(args.image, detections, os.path.join(args.output, "detections.png"))

    # 创建模拟眼动数据
    print(f"\n[*] Creating mock gaze data...")
    gaze_data = create_mock_gaze_data(centers, num_points=50)
    print(f"    Created {len(gaze_data)} gaze points")

    # 绘制热力图
    print(f"\n[*] Drawing heatmap...")
    plot_heatmap_on_image(args.image, gaze_data, os.path.join(args.output, "heatmap.png"))

    # 绘制轨迹
    print(f"\n[*] Drawing trajectory...")
    plot_trajectory_on_image(args.image, gaze_data, os.path.join(args.output, "trajectory.png"))

    print(f"\n[Done] All visualizations saved to: {args.output}/")
    print("\nGenerated files:")
    print("  - detections.png: 检测框和中心点")
    print("  - heatmap.png: 眼动热力图")
    print("  - trajectory.png: 眼动轨迹图")


if __name__ == "__main__":
    main()
