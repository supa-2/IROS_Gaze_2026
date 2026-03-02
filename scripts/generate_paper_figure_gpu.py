#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成论文用图表 - IROS Gaze 系统 (GPU服务器版本)
"""

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter
import torch
import cv2

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sam2_path = os.path.join(project_root, 'sam2')
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)


class SAM2Segmenter:
    """SAM2 自动分割器"""

    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device

        # 获取绝对路径（在切换目录之前）
        if not os.path.isabs(model_path):
            abs_model_path = os.path.join(project_root, model_path)
        else:
            abs_model_path = model_path

        print(f"[*] 初始化 SAM2...")
        print(f"    模型: {model_path}")

        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(f"模型文件不存在: {abs_model_path}")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # 根据文件名确定配置
            # 注意: sam2_hiera_small.pt 是 SAM2 v1，用 sam2_hiera_s
            #       sam2.1_hiera_small.pt 是 SAM2 v2.1，用 sam2.1_hiera_s
            model_filename = os.path.basename(model_path).lower()
            if 'sam2.1' in model_filename:
                # SAM2 v2.1 模型
                if 'hiera_small' in model_filename:
                    config_name = "sam2.1_hiera_s"
                elif 'hiera_tiny' in model_filename:
                    config_name = "sam2.1_hiera_t"
                else:
                    config_name = "sam2.1_hiera_s"
            else:
                # SAM2 v1 模型 (默认)
                if 'hiera_small' in model_filename or 'small' in model_filename:
                    config_name = "sam2_hiera_s"
                elif 'hiera_tiny' in model_filename or 'tiny' in model_filename:
                    config_name = "sam2_hiera_t"
                elif 'hiera_large' in model_filename or 'large' in model_filename:
                    config_name = "sam2_hiera_l"
                elif 'hiera_base+' in model_filename or 'b+' in model_filename:
                    config_name = "sam2_hiera_b+"
                else:
                    config_name = "sam2_hiera_s"

            print(f"    配置: {config_name}")

            # 直接调用 build_sam2，不切换目录
            model = build_sam2(
                config_file=config_name,
                ckpt_path=abs_model_path,
                device=device
            )

            self.predictor = SAM2ImagePredictor(model, device=device)
            print("[+] SAM2 加载成功")

        except Exception as e:
            print(f"[!] SAM2 加载失败: {e}")
            print("[*] 请检查:")
            print("    1. 模型文件存在: ls -la models/sam2/")
            print("    2. SAM2 已正确安装: cd sam2 && pip install -e .")
            raise


    def auto_segment_all(self, image_path, points_per_side=32,
                        pred_iou_thresh=0.88, stability_score_thresh=0.95,
                        min_mask_region_area=500, max_mask_region_area=500000):
        """自动分割图像中的所有区域"""
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        height, width = image_np.shape[:2]

        print(f"\n[*] 运行 SAM2 自动分割...")
        print(f"    图像尺寸: {width}x{height}")

        mask_generator = SAM2AutomaticMaskGenerator(
            model=self.predictor.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            max_mask_region_area=max_mask_region_area,
            output_mode='binary_mask'
        )

        masks = mask_generator.generate(image_np)
        print(f"[+] 发现 {len(masks)} 个区域")

        valid_masks = []
        for mask_data in masks:
            area = mask_data.get('area', 0)
            if min_mask_region_area <= area <= max_mask_region_area:
                score = mask_data.get('predicted_iou', 0)
                if score > pred_iou_thresh:
                    valid_masks.append(mask_data)

        valid_masks.sort(key=lambda x: x.get('area', 0), reverse=True)
        print(f"[+] 过滤后有效区域: {len(valid_masks)}")

        return valid_masks, image_np

    def create_segmentation_visualization(self, image_np, masks_data, output_path):
        """创建分割掩码可视化 - 分割区域显示原图，其他区域黑色"""
        height, width = image_np.shape[:2]

        # 创建黑色背景
        result = np.zeros_like(image_np)

        # 创建统一的掩码（所有分割区域）
        combined_mask = np.zeros((height, width), dtype=bool)

        for mask_data in masks_data:
            mask = mask_data.get('segmentation')
            combined_mask = combined_mask | mask

        # 只在掩码区域显示原图
        result[combined_mask] = image_np[combined_mask]

        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(result)
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        plt.close()
        print(f"[+] 保存 SAM2 分割图: {output_path}")

        return result


def predict_saliency_heatmap(image_path, masks_data, output_path, sigma=20):
    """基于视觉显著性模型预测热力图 - 只在分割区域显示热度"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    print("\n[*] 计算视觉显著性...")

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)

    g1 = cv2.GaussianBlur(l_channel, (0, 0), 5)
    g2 = cv2.GaussianBlur(l_channel, (0, 0), 20)
    brightness_contrast = np.abs(g1 - g2)

    a_blur = cv2.GaussianBlur(a_channel, (0, 0), 10)
    b_blur = cv2.GaussianBlur(b_channel, (0, 0), 10)
    color_contrast = np.sqrt((a_channel - a_blur) ** 2 + (b_channel - b_blur) ** 2)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 5)
    edges = edges / edges.max() if edges.max() > 0 else edges

    cy, cx = height // 2, width // 2
    y, x = np.mgrid[:height, :width]
    center_bias = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (min(height, width) / 3) ** 2))

    saliency = (
        brightness_contrast * 0.3 +
        color_contrast * 0.3 +
        edges * 0.2 +
        center_bias * 0.2
    )

    saliency = gaussian_filter(saliency, sigma=sigma)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    # 创建统一掩码
    combined_mask = np.zeros((height, width), dtype=bool)
    for mask_data in masks_data:
        mask = mask_data.get('segmentation')
        combined_mask = combined_mask | mask

    # 只在分割区域保留显著性
    saliency_masked = saliency.copy()
    saliency_masked[~combined_mask] = 0

    # 重新归一化（只在分割区域内）
    if saliency_masked.max() > 0:
        saliency_masked = saliency_masked / saliency_masked.max()

    # 使用 'hot' 色图（红黄色，适合热力图）
    colormap = plt.get_cmap('hot')
    colored_heatmap = colormap(saliency_masked)

    # 叠加到原图
    alpha = 0.7  # 热力图透明度
    result_array = image.copy().astype(np.float32)

    # 只在有显著性的区域叠加
    mask = saliency_masked > 0.1
    for c in range(3):
        result_array[:, :, c] = (
            image[:, :, c] * (1 - alpha * saliency_masked) +
            colored_heatmap[:, :, c] * 255 * alpha * saliency_masked
        )

    result_array = np.clip(result_array, 0, 255).astype(np.uint8)

    # 单图显示
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(result_array)
    ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] 保存预测热力图: {output_path}")

    return saliency_masked


def predict_scan_path(image_path, saliency_map, masks_data, output_path, num_fixations=10):
    """基于显著性和分割结果预测扫描路径"""
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    print(f"\n[*] 预测扫描路径 ({num_fixations} 个注视点)...")

    mask_scores = []
    for i, mask_data in enumerate(masks_data):
        mask = mask_data.get('segmentation')
        bbox = mask_data.get('bbox', [])

        mean_saliency = saliency_map[mask].mean()

        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        img_cx, img_cy = width/2, height/2
        dist_to_center = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
        center_bias = np.exp(-dist_to_center / (min(width, height) / 2))

        score = mean_saliency * 0.7 + center_bias * 0.3

        mask_scores.append({
            'index': i,
            'bbox': bbox,
            'center': (int(cx), int(cy)),
            'score': score,
            'area': mask_data.get('area', 0)
        })

    mask_scores.sort(key=lambda x: x['score'], reverse=True)
    fixations = mask_scores[:num_fixations]

    def scan_order_key(fix):
        cx, cy = fix['center']
        return cx * 0.7 + cy * 0.3

    fixations.sort(key=scan_order_key)

    for i, fix in enumerate(fixations):
        fix['sequence'] = i + 1
        base_duration = 30
        duration = base_duration * (0.5 + fix['score']) * (1 + np.log(fix['area'] / 1000 + 1) * 0.2)
        fix['duration'] = min(duration, 200)

    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(img_array)

    # 白色路径线
    if len(fixations) > 1:
        path_x = [f['center'][0] for f in fixations]
        path_y = [f['center'][1] for f in fixations]
        # 外层黑色轮廓（增加对比度）
        ax.plot(path_x, path_y, color='black', linewidth=5, alpha=0.8,
               linestyle='-', marker='', zorder=2)
        # 内层白色
        ax.plot(path_x, path_y, color='white', linewidth=3, alpha=1.0,
               linestyle='-', marker='', zorder=3)

    # 注视点圆圈
    for fix in fixations:
        cx, cy = fix['center']
        duration = fix['duration']
        seq = fix['sequence']

        radius = max(15, min(45, int(duration / 4)))

        # 黑色轮廓
        circle = Circle((cx, cy), radius, facecolor='white',
                       edgecolor='black', linewidth=4, alpha=0.95, zorder=4)
        ax.add_patch(circle)
        # 白色填充
        circle_inner = Circle((cx, cy), radius - 2, facecolor='white',
                       edgecolor='white', linewidth=2, alpha=0.9, zorder=5)
        ax.add_patch(circle_inner)

        # 序号
        ax.text(cx, cy, str(seq), color='black', fontsize=14, fontweight='bold',
               ha='center', va='center', zorder=6)

    ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] 保存预测轨迹图: {output_path}")

    return fixations


def generate_table_data(masks_data, fixations, image_path, output_dir):
    """生成用于论文表格的统计数据"""
    img = Image.open(image_path)
    width, height = img.size

    # 计算统计数据
    exhibits = []
    for i, mask_data in enumerate(masks_data[:15]):  # 最多15个展品
        bbox = mask_data.get('bbox', [])
        area = mask_data.get('area', 0)
        score = mask_data.get('predicted_iou', 0)

        x, y, w, h = bbox
        cx, cy = int(x + w/2), int(y + h/2)

        # 计算该展品的注视次数和总时长
        gaze_count = 0
        total_duration = 0
        first_fixation = None
        last_fixation = None

        for fix in fixations:
            if fix['index'] == i:
                gaze_count += 1
                total_duration += fix['duration']
                if first_fixation is None:
                    first_fixation = fix['sequence']
                last_fixation = fix['sequence']

        exhibits.append({
            'id': f'EX-{i:03d}',
            'bbox': [int(x) for x in bbox],
            'center': [cx, cy],
            'area_pixels': area,
            'area_ratio': f'{area / (width * height) * 100:.2f}%',
            'sam_score': f'{score:.3f}',
            'gaze_count': gaze_count,
            'total_duration': f'{total_duration:.1f}',
            'avg_duration': f'{total_duration / gaze_count:.1f}' if gaze_count > 0 else '0',
            'first_look': first_fixation,
            'last_look': last_fixation
        })

    summary = {
        'timestamp': '2026-03-02T00:00:00',
        'image_info': {
            'path': image_path,
            'width': width,
            'height': height,
            'total_pixels': width * height
        },
        'summary': {
            'total_exhibits': len(masks_data),
            'analyzed_exhibits': len(exhibits),
            'total_fixations': len(fixations),
            'gazed_exhibits': sum(1 for e in exhibits if e['gaze_count'] > 0)
        },
        'exhibits': exhibits,
        'fixations': [
            {
                'sequence': f['sequence'],
                'exhibit_id': f"EX-{f['index']:03d}",
                'center': f['center'],
                'duration': f'{f["duration"]:.1f}',
                'score': f'{f["score"]:.3f}'
            }
            for f in fixations
        ]
    }

    # 保存JSON
    json_path = os.path.join(output_dir, 'table_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[+] 保存表格数据: {json_path}")

    # 打印Markdown表格
    print("\n" + "=" * 100)
    print("论文表格数据 (Markdown格式)")
    print("=" * 100)
    print("| 展品ID | 边界框 | 面积占比 | SAM分数 | 注视次数 | 总时长(ms) | 平均时长 | 首次注视 |")
    print("|--------|--------|----------|---------|----------|------------|----------|----------|")
    for e in exhibits:
        bbox_str = f"({e['bbox'][0]},{e['bbox'][1]},{e['bbox'][2]},{e['bbox'][3]})"
        print(f"| {e['id']} | {bbox_str} | {e['area_ratio']} | {e['sam_score']} | {e['gaze_count']} | {e['total_duration']} | {e['avg_duration']} | {e['first_look'] or '-'} |")

    return summary


def create_paper_figure(image_path, output_path, sam2_model_path,
                       num_fixations=10, points_per_side=32):
    """生成论文用四宫格图表"""

    if not torch.cuda.is_available():
        print("[!] CUDA 不可用")
        return False

    print(f"\n[*] 处理图像: {image_path}")
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    original_img = Image.open(image_path).convert('RGB')
    img_array = np.array(original_img)
    width, height = original_img.size
    print(f"    尺寸: {width}x{height}")

    print("\n" + "="*60)
    print("步骤 (b): SAM2 自动分割")
    print("="*60)

    segmenter = SAM2Segmenter(sam2_model_path)
    masks_data, _ = segmenter.auto_segment_all(image_path, points_per_side)

    mask_path = output_path.replace('.png', '_mask.png')
    mask_result = segmenter.create_segmentation_visualization(img_array, masks_data, mask_path)

    print("\n" + "="*60)
    print("步骤 (c): 视觉显著性预测热力图")
    print("="*60)

    heatmap_path = output_path.replace('.png', '_heatmap.png')
    saliency_map = predict_saliency_heatmap(image_path, masks_data, heatmap_path)

    print("\n" + "="*60)
    print("步骤 (d): 扫描路径预测")
    print("="*60)

    trajectory_path = output_path.replace('.png', '_trajectory.png')
    fixations = predict_scan_path(image_path, saliency_map, masks_data, trajectory_path, num_fixations)

    print("\n" + "="*60)
    print("生成表格数据")
    print("="*60)

    generate_table_data(masks_data, fixations, image_path, output_dir)

    print("\n" + "="*60)
    print("生成四宫格图表")
    print("="*60)

    # 读取生成的图片
    mask_img = Image.open(mask_path)
    heatmap_img = Image.open(heatmap_path)
    trajectory_img = Image.open(trajectory_path)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    axes[0].imshow(original_img)
    axes[0].set_title('(a) Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask_img)
    axes[1].set_title('(b) Segmentation', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(heatmap_img)
    axes[2].set_title('(c) Heatmap', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(trajectory_img)
    axes[3].set_title('(d) Scan Path', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[+] 保存四宫格图表: {output_path}")

    print("\n" + "=" * 80)
    print("分割区域统计")
    print("=" * 80)
    print(f"{'ID':<5} {'Area':<10} {'Score':<10}")
    print("-" * 80)
    for i, mask_data in enumerate(masks_data[:15]):
        area = mask_data.get('area', 0)
        score = mask_data.get('predicted_iou', 0)
        print(f"{i+1:<5} {area:<10} {score:.3f}")
    print(f"\n共发现 {len(masks_data)} 个区域")

    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/outputs/paper_figure.png')
    parser.add_argument('--sam2-model', type=str, default='models/sam2/sam2_hiera_small.pt')
    parser.add_argument('--num-fixations', type=int, default=10)
    parser.add_argument('--points-per-side', type=int, default=32)

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] 图像不存在: {args.image}")
        return

    if torch.cuda.is_available():
        print(f"[+] CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"    显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("[!] CUDA 不可用")
        return

    print("=" * 60)
    print("IROS Gaze 论文图表生成")
    print("=" * 60)
    print(f"原图: {args.image}")
    print(f"模型: {args.sam2_model}")

    create_paper_figure(
        args.image, args.output, args.sam2_model,
        args.num_fixations, args.points_per_side
    )

    print("\n[OK] 完成!")


if __name__ == '__main__':
    main()
