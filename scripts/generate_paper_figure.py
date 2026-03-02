#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成论文用图表 - IROS Gaze 系统

生成四宫格图表：
- (a) 原图
- (b) 分割掩码
- (c) 热力图（基于眼动数据）
- (d) 轨迹图（基于眼动数据）
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter

# Times New Roman 字体
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['axes.unicode_minus'] = False

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入可视化模块
try:
    from skills.visualization.pixel_heatmap import GazeHeatmapVisualizer, FixationPoint
    from skills.visualization.gaze_trajectory import GazeTrajectoryVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"[!] 可视化模块导入失败: {e}")
    VISUALIZATION_AVAILABLE = False


# ==================== 分割掩码生成 ====================

def create_segmentation_mask(image_path, exhibits, output_path):
    """
    生成分割掩码图 - 黑色背景 + 灰色圆形掩码

    Args:
        image_path: 原图路径
        exhibits: 展品列表
        output_path: 输出路径
    """
    # 读取原图获取尺寸
    img = Image.open(image_path)
    width, height = img.size

    # 创建黑色背景
    mask_img = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask_img)

    # 为每个展品绘制圆形掩码
    for ex in exhibits:
        bbox = ex.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = min((x2 - x1) // 2, (y2 - y1) // 2)

            # 使用不同灰度值区分不同展品
            gray_value = 100 + (hash(ex.get('id', '')) % 100)
            color = (gray_value, gray_value, gray_value)

            # 绘制填充圆
            draw.ellipse([center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius],
                        fill=color, outline=(150, 150, 150), width=2)

    # 保存
    mask_img.save(output_path)
    print(f"[+] 保存分割掩码图: {output_path}")
    return mask_img


def create_sam2_segmentation(image_path, exhibits, output_path):
    """
    使用 SAM2 进行精细分割

    Args:
        image_path: 原图路径
        exhibits: 展品列表（提供 bbox 作为 prompt）
        output_path: 输出路径
    """
    try:
        from skills.segmentation.sam2_local import SAM2LocalSegmenter
        import torch

        # 检查 CUDA 可用性
        if not torch.cuda.is_available():
            print("[!] CUDA 不可用，使用圆形掩码代替")
            return create_segmentation_mask(image_path, exhibits, output_path)

        print("[*] 初始化 SAM2...")
        segmenter = SAM2LocalSegmenter(model_size='small', device='cuda')

        # 读取图像
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # 创建掩码叠加图
        fig, ax = plt.subplots(figsize=(img.size[0]/100, img.size[1]/100))
        ax.imshow(img_array)

        # 为每个展品进行分割
        colors = plt.cm.tab10(np.linspace(0, 1, len(exhibits)))

        for i, ex in enumerate(exhibits):
            bbox = ex.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox

                # 使用 bbox prompt 进行分割
                segmenter.set_image(img_array)
                masks, scores, _ = segmenter.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=True
                )

                if len(masks) > 0:
                    # 选择最佳掩码
                    best_mask = masks[0]

                    # 叠加显示掩码
                    color = colors[i]
                    colored_mask = np.zeros((*best_mask.shape, 4))
                    colored_mask[best_mask] = [*color[:3], 0.5]  # 半透明
                    ax.imshow(colored_mask)

        ax.set_title('(b)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[+] 保存 SAM2 分割掩码图: {output_path}")

    except Exception as e:
        print(f"[!] SAM2 分割失败: {e}")
        print("[*] 使用圆形掩码代替")
        return create_segmentation_mask(image_path, exhibits, output_path)


# ==================== 热力图生成 ====================

def create_heatmap_on_image(image_path, gaze_records, exhibits, output_path, sigma=50):
    """
    在原图上生成热力图

    Args:
        image_path: 原图路径
        gaze_records: 眼动记录列表
        exhibits: 展品列表
        output_path: 输出路径
        sigma: 高斯模糊标准差
    """
    # 读取原图
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    # 创建热力图数组
    heatmap = np.zeros((height, width))

    # 为展品创建中心点映射
    exhibit_centers = {}
    for ex in exhibits:
        bbox = ex.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            exhibit_centers[ex.get('id', '')] = (center_x, center_y)

    # 添加每个注视点的贡献
    for record in gaze_records:
        x = int(record.get('x', 0))
        y = int(record.get('y', 0))
        duration = record.get('duration', 0)

        # 如果坐标在展品边界内，使用展品中心
        matched = False
        for ex in exhibits:
            bbox = ex.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                if x1 <= x <= x2 and y1 <= y <= y2:
                    x, y = (x1 + x2) // 2, (y1 + y2) // 2
                    matched = True
                    break

        if 0 <= x < width and 0 <= y < height:
            # 创建单点热力图
            point_heatmap = np.zeros((height, width))
            point_heatmap[y, x] = duration

            # 高斯模糊
            blurred = gaussian_filter(point_heatmap, sigma=sigma)
            heatmap += blurred

    # 归一化
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # 创建颜色映射
    colormap = plt.get_cmap('jet')
    colored_heatmap = colormap(heatmap)

    # 叠加到原图
    alpha = 0.6
    result_array = img_array * (1 - alpha) + colored_heatmap[:, :, :3] * 255 * alpha
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)

    # 保存
    result_img = Image.fromarray(result_array)
    result_img.save(output_path)
    print(f"[+] 保存热力图: {output_path}")

    # 绘制带标记的热力图（在展品中心画点）
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(result_array)

    # 在每个展品中心画点
    for ex in exhibits:
        bbox = ex.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            ax.plot(center_x, center_y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_with_dots.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return heatmap


# ==================== 轨迹图生成 ====================

def create_trajectory_on_image(image_path, gaze_records, exhibits, output_path):
    """
    在原图上生成轨迹图

    Args:
        image_path: 原图路径
        gaze_records: 眼动记录列表（按sequence排序）
        exhibits: 展品列表
        output_path: 输出路径
    """
    # 读取原图
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    width, height = img.size

    # 创建展品中心映射
    exhibit_centers = {}
    for ex in exhibits:
        bbox = ex.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            exhibit_centers[ex.get('id', '')] = ((x1 + x2) // 2, (y1 + y2) // 2)

    # 按序列排序并转换坐标到展品中心
    sorted_records = sorted(gaze_records, key=lambda x: x.get('sequence', 0))

    # 创建轨迹点列表（使用展品中心）
    trajectory_points = []
    for record in sorted_records:
        ex_id = record.get('exhibit_id', '')
        if ex_id in exhibit_centers:
            x, y = exhibit_centers[ex_id]
            duration = record.get('duration', 0)
            sequence = record.get('sequence', 0)
            trajectory_points.append({'x': x, 'y': y, 'duration': duration, 'sequence': sequence})

    # 使用 matplotlib 绘制轨迹
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(img_array)

    # 绘制连线
    if len(trajectory_points) > 1:
        x_coords = [p['x'] for p in trajectory_points]
        y_coords = [p['y'] for p in trajectory_points]
        ax.plot(x_coords, y_coords, color='yellow', linewidth=2, alpha=0.8, zorder=1)

    # 绘制点和编号
    for i, p in enumerate(trajectory_points):
        x, y = p['x'], p['y']
        duration = p['duration']
        seq = p['sequence']

        # 根据时长确定圆点大小
        radius = max(10, min(30, int(duration / 5)))

        # 绘制圆点
        circle = MplCircle((x, y), radius, facecolor='orange', edgecolor='white', linewidth=2, alpha=0.8, zorder=2)
        ax.add_patch(circle)

        # 绘制编号
        ax.text(x, y, str(seq), color='black', fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=3)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[+] 保存轨迹图: {output_path}")
    plt.close()


# ==================== 四宫格图表生成 ====================

def create_paper_figure(
    original_image_path,
    json_path,
    output_path,
    use_sam2=False
):
    """
    生成论文用四宫格图表
    """
    # 读取数据
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    exhibits = json_data.get('exhibits', [])
    gaze_records = json_data.get('gaze_records', [])

    # 读取图片
    original_img = Image.open(original_image_path).convert("RGB")
    img_array = np.array(original_img)
    width, height = original_img.size

    print(f"\n[*] 处理图像: {original_image_path}")
    print(f"    尺寸: {width}x{height}")
    print(f"    展品数: {len(exhibits)}")
    print(f"    眼动记录: {len(gaze_records)}")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 生成分割掩码图
    print("\n[*] 生成分割掩码图...")
    mask_path = output_path.replace('.png', '_mask.png')
    if use_sam2:
        create_sam2_segmentation(original_image_path, exhibits, mask_path)
    else:
        create_segmentation_mask(original_image_path, exhibits, mask_path)
    mask_img = Image.open(mask_path)

    # 生成热力图
    print("\n[*] 生成热力图...")
    heatmap_path = output_path.replace('.png', '_heatmap.png')
    create_heatmap_on_image(original_image_path, gaze_records, exhibits, heatmap_path)
    heatmap_img = Image.open(heatmap_path.replace('.png', '_with_dots.png'))

    # 生成轨迹图
    print("\n[*] 生成轨迹图...")
    trajectory_path = output_path.replace('.png', '_trajectory.png')
    create_trajectory_on_image(original_image_path, gaze_records, exhibits, trajectory_path)
    trajectory_img = Image.open(trajectory_path)

    # 创建四宫格图表
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # (a) 原图
    axes[0].imshow(original_img)
    axes[0].set_title('(a)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # (b) 分割掩码图
    axes[1].imshow(mask_img)
    axes[1].set_title('(b)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # (c) 热力图
    axes[2].imshow(heatmap_img)
    axes[2].set_title('(c)', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # (d) 轨迹图
    axes[3].imshow(trajectory_img)
    axes[3].set_title('(d)', fontsize=14, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02)

    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[+] 保存四宫格图表: {output_path}")

    print(f"[+] 热力图: {heatmap_path}")
    print(f"[+] 轨迹图: {trajectory_path}")

    # 打印展品统计
    print("\n" + "=" * 80)
    print("展品列表")
    print("=" * 80)
    print(f"{'ID':<10} {'Name':<25} {'Type':<10}")
    print("-" * 80)
    for ex in exhibits:
        print(f"{ex.get('id', ''):<10} {ex.get('name', ''):<25} {ex.get('type', ''):<10}")

    plt.close()


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='生成论文用图表')
    parser.add_argument('--image', type=str, required=True, help='原图路径')
    parser.add_argument('--json', type=str,
                       default='data/outputs/pipeline_vlm/FINAL_REPORT.json',
                       help='FINAL_REPORT.json 路径')
    parser.add_argument('--output', type=str,
                       default='data/outputs/paper_figure.png',
                       help='输出路径')
    parser.add_argument('--use-sam2', action='store_true',
                       help='使用 SAM2 进行精细分割（需要 GPU）')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] 图像不存在: {args.image}")
        return

    if not os.path.exists(args.json):
        print(f"[!] JSON 不存在: {args.json}")
        return

    print("=" * 60)
    print("IROS Gaze 论文图表生成")
    print("=" * 60)
    print(f"原图: {args.image}")
    print(f"JSON: {args.json}")
    print(f"输出: {args.output}")
    print(f"SAM2: {'是' if args.use_sam2 else '否'}")

    create_paper_figure(
        original_image_path=args.image,
        json_path=args.json,
        output_path=args.output,
        use_sam2=args.use_sam2
    )

    print("\n[OK] 完成!")


if __name__ == '__main__':
    main()
