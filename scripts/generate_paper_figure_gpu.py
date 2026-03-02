#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成论文用图表 - IROS Gaze 系统 (GPU服务器版本)
直接从checkpoint加载SAM2，不依赖hydra
"""

import os
import sys
import argparse
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
    """SAM2 自动分割器 - 直接从 checkpoint 加载"""

    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device

        print(f"[*] 初始化 SAM2...")
        print(f"    模型: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 使用SAM2的官方API - build_sam2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # 根据模型文件名确定配置
        model_filename = os.path.basename(model_path)
        if "sam2.1_hiera_small.pt" in model_filename or "sam2_hiera_small.pt" in model_filename:
            config_name = "sam2.1_hiera_s"
        elif "sam2.1_hiera_tiny.pt" in model_filename or "sam2_hiera_tiny.pt" in model_filename:
            config_name = "sam2.1_hiera_t"
        elif "sam2.1_hiera_large.pt" in model_filename or "sam2_hiera_large.pt" in model_filename:
            config_name = "sam2.1_hiera_l"
        elif "sam2.1_hiera_base_plus.pt" in model_filename or "sam2_hiera_base_plus.pt" in model_filename:
            config_name = "sam2.1_hiera_b+"
        elif "sam2_hiera_small.pt" in model_filename:
            config_name = "sam2_hiera_s"
        else:
            # 默认使用 sam2.1_hiera_s
            config_name = "sam2.1_hiera_s"
            print(f"    [警告] 无法从文件名推断配置，使用默认: {config_name}")

        print(f"    配置: {config_name}")

        # 使用 hydra_overrides 直接指定 checkpoint 路径
        hydra_overrides = [f"+ckpt_path={os.path.abspath(model_path)}"]

        # 构建模型
        model = build_sam2(
            config_file=config_name,
            ckpt_path=os.path.abspath(model_path),
            device=device,
            hydra_overrides_extra=hydra_overrides
        )

        self.predictor = SAM2ImagePredictor(model, device=device)
        print("[+] SAM2 加载成功")

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
        """创建分割掩码可视化"""
        height, width = image_np.shape[:2]

        # 黑色背景版本
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.set_facecolor('black')

        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(masks_data))))

        for i, mask_data in enumerate(masks_data):
            mask = mask_data.get('segmentation')
            color = colors[i % 10]

            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*color[:3], 0.85]
            ax.imshow(colored_mask, interpolation='none')

            bbox = mask_data.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                ax.text(x, y-10 if y > 20 else y+h+20, str(i+1),
                       color='white', fontsize=max(8, min(12, w//5)),
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='black', edgecolor='white', alpha=0.8))

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        plt.close()
        print(f"[+] 保存 SAM2 分割图: {output_path}")

        # 轮廓版本
        output_outline = output_path.replace('.png', '_outline.png')
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(image_np)

        for i, mask_data in enumerate(masks_data):
            mask = mask_data.get('segmentation')
            color = colors[i % 10]

            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                for j in range(len(approx)):
                    pt1 = tuple(approx[j][0].tolist())
                    pt2 = tuple(approx[(j+1) % len(approx)][0].tolist())
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                           color=color, linewidth=2, alpha=0.8)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_outline, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[+] 保存轮廓图: {output_outline}")


def predict_saliency_heatmap(image_path, output_path, sigma=30):
    """基于视觉显著性模型预测热力图"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    print("\n[*] 计算视觉显著性...")

    # Lab 颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)

    # 亮度对比
    g1 = cv2.GaussianBlur(l_channel, (0, 0), 5)
    g2 = cv2.GaussianBlur(l_channel, (0, 0), 20)
    brightness_contrast = np.abs(g1 - g2)

    # 颜色对比
    a_blur = cv2.GaussianBlur(a_channel, (0, 0), 10)
    b_blur = cv2.GaussianBlur(b_channel, (0, 0), 10)
    color_contrast = np.sqrt((a_channel - a_blur) ** 2 + (b_channel - b_blur) ** 2)

    # 边缘密度
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 5)
    edges = edges / edges.max() if edges.max() > 0 else edges

    # 中心偏置
    cy, cx = height // 2, width // 2
    y, x = np.mgrid[:height, :width]
    center_bias = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (min(height, width) / 3) ** 2))

    # 组合
    saliency = (
        brightness_contrast * 0.3 +
        color_contrast * 0.3 +
        edges * 0.2 +
        center_bias * 0.2
    )

    saliency = gaussian_filter(saliency, sigma=sigma)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    # 叠加到原图
    img_pil = Image.open(image_path).convert('RGB')
    img_array = np.array(img_pil)

    colormap = plt.get_cmap('jet')
    colored_heatmap = colormap(saliency)

    alpha = 0.6
    result_array = img_array * (1 - alpha) + colored_heatmap[:, :, :3] * 255 * alpha
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)

    # 保存
    fig, axes = plt.subplots(1, 2, figsize=(width/100 + 3, height/100))
    axes[0].imshow(img_array)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(result_array)
    axes[1].set_title('Predicted Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] 保存预测热力图: {output_path}")

    return saliency


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

    # 绘制
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(img_array)

    # 绘制轮廓
    colors = plt.cm.tab10(np.linspace(0, 1, len(masks_data)))
    for i, mask_data in enumerate(masks_data[:15]):
        mask = mask_data.get('segmentation')
        color = colors[i % 10]

        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.003 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) > 2:
                pts = approx.squeeze()
                if len(pts.shape) == 2:
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1, alpha=0.4)

    # 绘制扫描路径
    if len(fixations) > 1:
        path_x = [f['center'][0] for f in fixations]
        path_y = [f['center'][1] for f in fixations]
        ax.plot(path_x, path_y, color='yellow', linewidth=3, alpha=0.9,
               linestyle='-', marker='', zorder=2)

    for fix in fixations:
        cx, cy = fix['center']
        duration = fix['duration']
        seq = fix['sequence']

        radius = max(15, min(45, int(duration / 4)))

        circle = Circle((cx, cy), radius, facecolor='orange',
                       edgecolor='white', linewidth=3, alpha=0.95, zorder=3)
        ax.add_patch(circle)

        ax.text(cx, cy, str(seq), color='black', fontsize=12, fontweight='bold',
               ha='center', va='center', zorder=4)

    ax.text(10, 20, f'Predicted Scan Path ({len(fixations)} fixations)',
           color='white', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

    ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] 保存预测轨迹图: {output_path}")

    return fixations


def create_paper_figure(image_path, output_path, sam2_model_path,
                       num_fixations=10, points_per_side=32):
    """生成论文用四宫格图表"""

    if not torch.cuda.is_available():
        print("[!] CUDA 不可用")
        return False

    print(f"\n[*] 处理图像: {image_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    original_img = Image.open(image_path).convert('RGB')
    img_array = np.array(original_img)
    width, height = original_img.size
    print(f"    尺寸: {width}x{height}")

    # SAM2 自动分割
    print("\n" + "="*60)
    print("步骤 (b): SAM2 自动分割")
    print("="*60)

    segmenter = SAM2Segmenter(sam2_model_path)
    masks_data, _ = segmenter.auto_segment_all(image_path, points_per_side)

    mask_path = output_path.replace('.png', '_mask.png')
    segmenter.create_segmentation_visualization(img_array, masks_data, mask_path)
    mask_img = Image.open(mask_path)

    # 预测热力图
    print("\n" + "="*60)
    print("步骤 (c): 视觉显著性预测热力图")
    print("="*60)

    heatmap_path = output_path.replace('.png', '_heatmap.png')
    saliency_map = predict_saliency_heatmap(image_path, heatmap_path)
    heatmap_img = Image.open(heatmap_path)

    # 预测扫描路径
    print("\n" + "="*60)
    print("步骤 (d): 扫描路径预测")
    print("="*60)

    trajectory_path = output_path.replace('.png', '_trajectory.png')
    predict_scan_path(image_path, saliency_map, masks_data, trajectory_path, num_fixations)
    trajectory_img = Image.open(trajectory_path)

    # 创建四宫格图表
    print("\n" + "="*60)
    print("生成四宫格图表")
    print("="*60)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    axes[0].imshow(original_img)
    axes[0].set_title('(a)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask_img)
    axes[1].set_title('(b)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(heatmap_img)
    axes[2].set_title('(c)', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(trajectory_img)
    axes[3].set_title('(d)', fontsize=14, fontweight='bold')
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
