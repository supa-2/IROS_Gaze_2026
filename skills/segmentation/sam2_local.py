#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 2 本地分割模块

使用 Meta SAM 2 模型进行本地语义分割
无需调用 API，所有计算在本地完成
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch

# 添加 SAM 2 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sam2_path = os.path.join(project_root, 'sam2')
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError as e:
    print(f"[!] SAM 2 not available: {e}")
    SAM2_AVAILABLE = False


class SAM2LocalSegmenter:
    """
    SAM 2 本地分割器

    使用本地 SAM 2 模型进行图像分割
    """

    # 模型配置 - sam2.1 版本
    # 支持两种路径：models/sam2/ (项目根目录) 和 sam2/checkpoints/ (本地)
    MODELS = {
        'tiny': {
            'config': 'sam2/configs/sam2.1/sam2.1_hiera_t.yaml',
            'checkpoint': 'models/sam2/sam2.1_hiera_tiny.pt',
            'checkpoint_alt': 'sam2/checkpoints/sam2.1_hiera_tiny.pt',
            'size': 39  # MB
        },
        'small': {
            'config': 'sam2/configs/sam2.1/sam2.1_hiera_s.yaml',
            'checkpoint': 'models/sam2/sam2.1_hiera_small.pt',
            'checkpoint_alt': 'sam2/checkpoints/sam2.1_hiera_small.pt',
            'size': 46  # MB
        },
        'base_plus': {
            'config': 'sam2/configs/sam2.1/sam2.1_hiera_b+.yaml',
            'checkpoint': 'models/sam2/sam2.1_hiera_base_plus.pt',
            'checkpoint_alt': 'sam2/checkpoints/sam2.1_hiera_base_plus.pt',
            'size': 81  # MB
        },
        'large': {
            'config': 'sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
            'checkpoint': 'models/sam2/sam2.1_hiera_large.pt',
            'checkpoint_alt': 'sam2/checkpoints/sam2.1_hiera_large.pt',
            'size': 224  # MB
        }
    }

    def __init__(self, model_size: str = 'small', device: str = None):
        """
        初始化 SAM 2 分割器

        Args:
            model_size: 模型大小 ('tiny', 'small', 'base_plus', 'large')
            device: 设备 ('cuda', 'cpu', None=auto)
        """
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM 2 is not available. Please check installation.")

        self.model_size = model_size

        # 确定设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"[*] Initializing SAM 2 ({model_size}) on {self.device}...")

        # 获取模型配置
        model_config = self.MODELS.get(model_size, self.MODELS['small'])

        # 构建完整路径 - 尝试两个可能的位置
        checkpoint_path = os.path.join(project_root, model_config['checkpoint'])
        if not os.path.exists(checkpoint_path):
            checkpoint_path_alt = os.path.join(project_root, model_config.get('checkpoint_alt', ''))
            if os.path.exists(checkpoint_path_alt):
                checkpoint_path = checkpoint_path_alt
            else:
                raise FileNotFoundError(
                    f"Model checkpoint not found.\n"
                    f"  Tried: {checkpoint_path}\n"
                    f"  Tried: {checkpoint_path_alt}\n"
                    f"Please download using: python scripts/download_sam2_models.py --model {model_size}\n"
                    f"Or download from: https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
                )

        # 使用 build_sam2 直接加载
        from sam2.build_sam import build_sam2

        # 配置名称映射（使用完整路径）
        config_names = {
            'tiny': 'configs/sam2.1/sam2.1_hiera_t',
            'small': 'configs/sam2.1/sam2.1_hiera_s',
            'base_plus': 'configs/sam2.1/sam2.1_hiera_b+',
            'large': 'configs/sam2.1/sam2.1_hiera_l'
        }

        config_name = config_names[model_size]

        print(f"[*] Loading model from: {checkpoint_path}")
        print(f"[*] Config: {config_name}")

        # 构建模型
        model = build_sam2(
            config_file=config_name,
            ckpt_path=checkpoint_path,
            device=self.device
        )

        # 创建预测器
        self.predictor = SAM2ImagePredictor(model, device=self.device)

        print(f"[+] SAM 2 ({model_size}) loaded successfully")

    def set_image(self, image: np.ndarray):
        """
        设置要处理的图像

        Args:
            image: RGB 图像数组 (H, W, C)
        """
        self.predictor.set_image(image)

    def predict(
        self,
        point_coords: np.ndarray = None,
        point_labels: np.ndarray = None,
        box: np.ndarray = None,
        mask_input: np.ndarray = None,
        multimask_output: bool = True,
        return_logits: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测分割掩码

        Args:
            point_coords: 点坐标 (N, 2)
            point_labels: 点标签 (N,) 1=前景, 0=背景
            box: 边界框 [x1, y1, x2, y2]
            mask_input: 之前的掩码
            multimask_output: 是否输出多个掩码
            return_logits: 是否返回 logits

        Returns:
            masks: 掩码数组
            iou_scores: IoU 分数
            low_res_logits: 低分辨率 logits
        """
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits
        )

    def auto_segment(
        self,
        image: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.9,
        stability_score_thresh: float = 0.96,
        min_mask_region_area: int = 100,
        output_mode: str = 'binary_mask'
    ) -> List[Dict]:
        """
        自动生成所有掩码

        Args:
            image: 输入图像
            points_per_side: 每边的点数
            pred_iou_thresh: 预测 IoU 阈值
            stability_score_thresh: 稳定性分数阈值
            min_mask_region_area: 最小掩码区域面积
            output_mode: 输出模式

        Returns:
            掩码列表
        """
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        mask_generator = SAM2AutomaticMaskGenerator(
            model=self.predictor.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            output_mode=output_mode,
        )

        masks = mask_generator.generate(image)
        return masks

    def segment_exhibits(
        self,
        image_path: str,
        min_area: int = 5000,
        max_area: int = 500000
    ) -> List[Dict]:
        """
        分割图像中的展品

        Args:
            image_path: 图像路径
            min_area: 最小区域面积
            max_area: 最大区域面积

        Returns:
            展品区域列表
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)

        print(f"[*] Running auto-segmentation on {image_path}...")
        masks = self.auto_segment(image_array)

        # 过滤结果
        exhibits = []
        width, height = image.size

        for i, mask_data in enumerate(masks):
            area = mask_data.get('area', 0)
            bbox = mask_data.get('bbox', [0, 0, 0, 0])  # x1, y1, w, h

            # 转换 bbox 格式
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            # 过滤条件
            if min_area <= area <= max_area:
                # 检查是否在图像范围内
                if 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height:
                    exhibits.append({
                        'id': f'EX-{i:03d}',
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'area': area,
                        'score': mask_data.get('predicted_iou', 0.0),
                        'mask': mask_data.get('segmentation')
                    })

        print(f"[+] Found {len(exhibits)} exhibit regions")

        return exhibits


# ============================================
# 测试和演示
# ============================================

def test_sam2_local():
    """测试 SAM 2 本地分割"""
    import matplotlib.pyplot as plt

    print("\n" + "="*70)
    print("SAM 2 本地分割测试")
    print("="*70)

    # 初始化分割器
    try:
        segmenter = SAM2LocalSegmenter(model_size='small')
    except Exception as e:
        print(f"[!] 初始化失败: {e}")
        print("\n提示: 请确保已下载模型文件到 sam2/checkpoints/ 目录")
        return

    # 测试图像
    image_path = "data/R.jpg"

    if not os.path.exists(image_path):
        print(f"[!] 测试图像不存在: {image_path}")
        return

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    print(f"\n[*] 测试图像: {image.size}")

    # 设置图像
    segmenter.set_image(image_array)

    # 测试点预测（使用图像中心点）
    center_x, center_y = image.size[0] // 2, image.size[1] // 2
    point_coords = np.array([[center_x, center_y]])
    point_labels = np.array([1])

    masks, scores, logits = segmenter.predict(
        point_coords=point_coords,
        point_labels=point_labels
    )

    print(f"[+] 点预测完成，生成了 {len(masks)} 个掩码")

    # 自动分割
    print("\n[*] 运行自动分割...")
    auto_masks = segmenter.auto_segment(image_array)
    print(f"[+] 自动分割完成，找到了 {len(auto_masks)} 个区域")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    axes[0].imshow(image_array)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 点预测掩码
    axes[1].imshow(image_array)
    if len(masks) > 0:
        axes[1].imshow(masks[0], alpha=0.5)
    axes[1].scatter([center_x], [center_y], c='red', s=100)
    axes[1].set_title("Point Prediction")
    axes[1].axis('off')

    # 自动分割
    axes[2].imshow(image_array)
    for mask_data in auto_masks[:10]:  # 只显示前10个
        mask = mask_data.get('segmentation')
        if mask is not None:
            color = np.random.rand(3)
            axes[2].imshow(mask, alpha=0.3 * np.ones((*mask.shape, 1)) * color)
    axes[2].set_title(f"Auto Segmentation ({len(auto_masks)} regions)")
    axes[2].axis('off')

    plt.tight_layout()

    output_dir = "data/outputs/sam2_local"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[+] 结果已保存到: {output_path}")
    plt.close()


def segment_with_sam2_local(image_path: str, output_dir: str = None) -> List[Dict]:
    """
    使用本地 SAM 2 分割图像

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录

    Returns:
        展品区域列表
    """
    if output_dir is None:
        output_dir = "data/outputs/sam2_local"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化分割器
    segmenter = SAM2LocalSegmenter(model_size='small')

    # 分割展品
    exhibits = segmenter.segment_exhibits(image_path)

    # 可视化结果
    image = Image.open(image_path).convert('RGB')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, exhibit in enumerate(exhibits):
        bbox = exhibit['bbox']
        color = colors[i % 10]

        # 绘制边界框
        rect = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        # 绘制标签
        ax.text(
            bbox[0], bbox[1] - 10,
            f"{i+1}",
            color=color,
            fontsize=12,
            fontweight='bold'
        )

    ax.set_title(f"SAM 2 Local Segmentation - {len(exhibits)} exhibits found")
    ax.axis('off')

    output_path = os.path.join(output_dir, "segmentation_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 保存 JSON
    import json
    json_path = os.path.join(output_dir, "exhibits.json")
    # 移除 mask 字段（无法序列化）
    exhibits_copy = [{k: v for k, v in ex.items() if k != 'mask'} for ex in exhibits]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(exhibits_copy, f, indent=2, ensure_ascii=False)

    print(f"[+] 结果已保存到: {output_dir}")

    return exhibits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM 2 本地分割")
    parser.add_argument("--image", type=str, default="data/R.jpg", help="输入图像")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--segment", action="store_true", help="运行分割")

    args = parser.parse_args()

    if args.test:
        test_sam2_local()
    elif args.segment:
        segment_with_sam2_local(args.image)
    else:
        test_sam2_local()
