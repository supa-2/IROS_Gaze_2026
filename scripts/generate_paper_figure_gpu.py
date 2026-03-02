#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成论文用图表 - IROS Gaze 系统 (GPU服务器版本)

生成四宫格图表：
- (a) 原图
- (b) SAM2 分割掩码（精细分割）
- (c) 热力图（基于眼动数据）
- (d) 轨迹图（基于眼动数据）

依赖:
- SAM2 模型
- CUDA GPU
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle, Rectangle
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter
import torch

# Times New Roman 字体
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['axes.unicode_minus'] = False

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 配置 SAM2 模型路径
SAM2_MODEL_PATH = os.environ.get('SAM2_MODEL_PATH', '/home/g/models/iros_agent/models/sam2/sam2_hiera_small.pt')
SAM2_CONFIG_PATH = os.environ.get('SAM2_CONFIG_PATH', 'sam2/configs/sam2.1/sam2.1_hiera_s.yaml')


# ==================== SAM2 分割 ====================

class SAM2Segmenter:
    """SAM2 分割器"""

    def __init__(self, model_path=None, config_path=None, device='cuda'):
        self.model_path = model_path or SAM2_MODEL_PATH
        self.config_path = config_path or SAM2_CONFIG_PATH
        self.device = device

        print(f"[*] 初始化 SAM2...")
        print(f"    模型: {self.model_path}")
        print(f"    配置: {self.config_path}")
        print(f"    设备: {self.device}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 添加 SAM2 路径
        sam2_path = os.path.join(project_root, 'sam2')
        if sam2_path not in sys.path:
            sys.path.insert(0, sam2_path)

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # 构建模型
            model = build_sam2(
                config_file=self.config_path,
                ckpt_path=self.model_path,
                device=self.device
            )
            self.predictor = SAM2ImagePredictor(model, device=self.device)
            print("[+] SAM2 加载成功")

        except ImportError as e:
            raise RuntimeError(f"SAM2 导入失败: {e}. 请确保已安装 SAM2: pip install git+https://github.com/facebookresearch/segment-anything-2.git")

    def segment_exhibits(self, image, bboxes):
        """
        使用 bbox prompt 分割展品

        Args:
            image: PIL Image 或 numpy array
            bboxes: 展品 bbox 列表 [[x1, y1, x2, y2], ...]

        Returns:
            分割掩码列表
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        self.predictor.set_image(image)

        masks = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox

            # 使用 bbox prompt 进行预测
            pred_masks, scores, _ = self.predictor.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=True
            )

            # 选择最佳掩码（分数最高的）
            if len(scores) > 0:
                best_idx = np.argmax(scores)
                masks.append({
                    'mask': pred_masks[best_idx],
                    'score': float(scores[best_idx]),
                    'bbox': bbox
                })
                print(f"    展品 {i+1}: IoU={scores[best_idx]:.3f}")

        return masks

    def create_segmentation_visualization(self, image, masks, exhibits, output_path):
        """
        创建分割掩码可视化

        Args:
            image: 原图
            masks: SAM2 分割掩码列表
            exhibits: 展品信息
            output_path: 输出路径
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        fig, ax = plt.subplots(figsize=(image.shape[1]/100, image.shape[0]/100))

        # 显示原图作为背景（灰度）
        gray_bg = np.mean(image, axis=2)
        ax.imshow(gray_bg, cmap='gray', alpha=0.3)

        # 为每个掩码使用不同颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(masks)))

        for i, (mask_data, ex) in enumerate(zip(masks, exhibits)):
            mask = mask_data['mask']

            # 创建彩色掩码
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*colors[i][:3], 0.7]  # 半透明

            ax.imshow(colored_mask)

            # 添加展品ID标签
            bbox = ex.get('bbox', mask_data['bbox'])
            x1, y1 = bbox[0], bbox[1]
            ax.text(x1, y1-15, ex.get('id', f'{i+1}'),
                   color='white', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        ax.set_title('(b)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[+] 保存 SAM2 分割图: {output_path}")


# ==================== 热力图生成 ====================

def create_heatmap_on_image(image_path, gaze_records, exhibits, output_path, sigma=50):
    """在原图上生成热力图"""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    # 创建热力图数组
    heatmap = np.zeros((height, width))

    # 添加每个注视点的贡献
    for record in gaze_records:
        x = int(record.get('x', 0))
        y = int(record.get('y', 0))
        duration = record.get('duration', 0)

        if 0 <= x < width and 0 <= y < height:
            point_heatmap = np.zeros((height, width))
            point_heatmap[y, x] = duration
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

    # 绘制带标记的热力图
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(result_array)

    # 在每个展品中心画点
    for ex in exhibits:
        bbox = ex.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            ax.plot(center_x, center_y, 'ro', markersize=8,
                   markeredgecolor='white', markeredgewidth=2)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] 保存热力图: {output_path}")

    return heatmap


# ==================== 轨迹图生成 ====================

def create_trajectory_on_image(image_path, gaze_records, exhibits, output_path):
    """在原图上生成轨迹图"""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    width, height = img.size

    # 按序列排序
    sorted_records = sorted(gaze_records, key=lambda x: x.get('sequence', 0))

    # 使用 matplotlib 绘制轨迹
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(img_array)

    # 绘制连线
    if len(sorted_records) > 1:
        x_coords = [r.get('x', 0) for r in sorted_records]
        y_coords = [r.get('y', 0) for r in sorted_records]
        ax.plot(x_coords, y_coords, color='yellow', linewidth=2.5, alpha=0.9, zorder=1)

    # 绘制点和编号
    for r in sorted_records:
        x, y = r.get('x', 0), r.get('y', 0)
        duration = r.get('duration', 0)
        seq = r.get('sequence', 0)

        # 根据时长确定圆点大小
        radius = max(15, min(40, int(duration / 3)))

        # 绘制圆点
        circle = MplCircle((x, y), radius, facecolor='orange',
                          edgecolor='white', linewidth=2, alpha=0.9, zorder=2)
        ax.add_patch(circle)

        # 绘制编号
        ax.text(x, y, str(seq), color='black', fontsize=11, fontweight='bold',
                ha='center', va='center', zorder=3)

    # 绘制展品边界框
    for ex in exhibits:
        bbox = ex.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor='cyan', facecolor='none', alpha=0.5, zorder=0)
            ax.add_patch(rect)

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
    use_sam2=True,
    sam2_model_path=None,
    sam2_config_path=None
):
    """生成论文用四宫格图表"""

    # 检查 CUDA
    if use_sam2 and not torch.cuda.is_available():
        print("[!] CUDA 不可用，将使用圆形掩码代替 SAM2")
        use_sam2 = False

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

    # ========== 生成分割掩码图 ==========
    print("\n[*] 生成分割掩码图...")
    mask_path = output_path.replace('.png', '_mask.png')

    if use_sam2:
        try:
            segmenter = SAM2Segmenter(
                model_path=sam2_model_path,
                config_path=sam2_config_path
            )

            # 获取所有 bbox
            bboxes = [ex['bbox'] for ex in exhibits]

            # 分割
            print("[*] 运行 SAM2 分割...")
            masks = segmenter.segment_exhibits(original_img, bboxes)

            # 创建可视化
            segmenter.create_segmentation_visualization(
                original_img, masks, exhibits, mask_path
            )
            mask_img = Image.open(mask_path)

        except Exception as e:
            print(f"[!] SAM2 分割失败: {e}")
            print("[*] 使用圆形掩码代替")
            use_sam2 = False

    if not use_sam2:
        # 使用圆形掩码
        mask_img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(mask_img)

        for i, ex in enumerate(exhibits):
            bbox = ex.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = min((x2 - x1) // 2, (y2 - y1) // 2)

                # 使用不同灰度值
                gray_value = 80 + (i * 30) % 150
                draw.ellipse([center_x - radius, center_y - radius,
                             center_x + radius, center_y + radius],
                            fill=(gray_value, gray_value, gray_value),
                            outline=(150, 150, 150), width=2)

        mask_img.save(mask_path)
        print(f"[+] 保存圆形掩码图: {mask_path}")

    # ========== 生成热力图 ==========
    print("\n[*] 生成热力图...")
    heatmap_path = output_path.replace('.png', '_heatmap.png')
    create_heatmap_on_image(original_image_path, gaze_records, exhibits, heatmap_path)
    heatmap_img = Image.open(heatmap_path)

    # ========== 生成轨迹图 ==========
    print("\n[*] 生成轨迹图...")
    trajectory_path = output_path.replace('.png', '_trajectory.png')
    create_trajectory_on_image(original_image_path, gaze_records, exhibits, trajectory_path)
    trajectory_img = Image.open(trajectory_path)

    # ========== 创建四宫格图表 ==========
    print("\n[*] 生成四宫格图表...")
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
    print(f"[+] 保存四宫格图表: {output_path}")

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
    parser = argparse.ArgumentParser(description='生成论文用图表 (GPU服务器版本)')
    parser.add_argument('--image', type=str, required=True, help='原图路径')
    parser.add_argument('--json', type=str,
                       default='data/outputs/pipeline_vlm/FINAL_REPORT_TEST1.json',
                       help='眼动数据JSON路径')
    parser.add_argument('--output', type=str,
                       default='data/outputs/paper_figure_gpu.png',
                       help='输出路径')
    parser.add_argument('--sam2-model', type=str,
                       default='/home/g/models/iros_agent/models/sam2/sam2_hiera_small.pt',
                       help='SAM2 模型路径')
    parser.add_argument('--sam2-config', type=str,
                       default='sam2/configs/sam2.1/sam2.1_hiera_s.yaml',
                       help='SAM2 配置路径')
    parser.add_argument('--no-sam2', action='store_true',
                       help='禁用 SAM2，使用圆形掩码')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] 图像不存在: {args.image}")
        return

    if not os.path.exists(args.json):
        print(f"[!] JSON 不存在: {args.json}")
        return

    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"[+] CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"    显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("[!] CUDA 不可用，SAM2 分割将被禁用")

    print("=" * 60)
    print("IROS Gaze 论文图表生成 (GPU服务器版本)")
    print("=" * 60)
    print(f"原图: {args.image}")
    print(f"JSON: {args.json}")
    print(f"输出: {args.output}")
    print(f"SAM2: {'否' if args.no_sam2 else '是'}")
    if not args.no_sam2:
        print(f"模型: {args.sam2_model}")

    create_paper_figure(
        original_image_path=args.image,
        json_path=args.json,
        output_path=args.output,
        use_sam2=not args.no_sam2,
        sam2_model_path=args.sam2_model,
        sam2_config_path=args.sam2_config
    )

    print("\n[OK] 完成!")


if __name__ == '__main__':
    main()
