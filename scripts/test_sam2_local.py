#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 2 本地分割测试脚本

测试本地部署的 SAM 2 模型
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from skills.segmentation.sam2_local import SAM2LocalSegmenter, segment_with_sam2_local


def visualize_segmentation(image_path: str, exhibits: list, output_path: str):
    """可视化分割结果"""
    image = Image.open(image_path).convert('RGB')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：原图
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 右图：分割结果
    axes[1].imshow(image)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, exhibit in enumerate(exhibits):
        bbox = exhibit['bbox']
        color = colors[i % 10]

        # 绘制边界框
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        axes[1].add_patch(rect)

        # 绘制标签
        axes[1].text(
            bbox[0], bbox[1] - 15,
            f"{i+1}. {exhibit['id']}",
            color=color,
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5)
        )

    axes[1].set_title(f"SAM 2 Segmentation - {len(exhibits)} exhibits", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[+] Visualization saved to: {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM 2 本地分割测试")
    parser.add_argument("--image", type=str, default="data/R.jpg", help="输入图像")
    parser.add_argument("--model", type=str, default="small", choices=['tiny', 'small', 'base_plus', 'large'], help="模型大小")
    parser.add_argument("--min-area", type=int, default=5000, help="最小区域面积")
    parser.add_argument("--output", type=str, default="data/outputs/sam2_local", help="输出目录")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] 图像不存在: {args.image}")
        return

    print("\n" + "="*70)
    print("SAM 2 本地分割测试")
    print("="*70)

    os.makedirs(args.output, exist_ok=True)

    # 初始化分割器
    print(f"\n[*] 初始化 SAM 2 ({args.model})...")
    try:
        segmenter = SAM2LocalSegmenter(model_size=args.model, device='cpu')
    except Exception as e:
        print(f"[!] 初始化失败: {e}")
        print("\n提示: 请确保已下载模型文件到 sam2/checkpoints/ 目录")
        print("可从以下地址下载: https://dl.fbaipublicfiles.com/segment_anything_2/092824/")
        return

    # 分割展品
    print(f"\n[*] 分割图像: {args.image}")
    exhibits = segmenter.segment_exhibits(
        args.image,
        min_area=args.min_area
    )

    print(f"\n[+] 找到 {len(exhibits)} 个展品区域")

    # 显示前几个结果
    print("\n前 5 个展品:")
    for i, ex in enumerate(exhibits[:5]):
        bbox = ex['bbox']
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        print(f"  {i+1}. {ex['id']}: 位置=({bbox[0]}, {bbox[1]}), 大小={w}x{h}, 面积={ex['area']:.0f}, 分数={ex['score']:.3f}")

    # 可视化
    output_path = os.path.join(args.output, "segmentation_result.png")
    visualize_segmentation(args.image, exhibits, output_path)

    # 保存 JSON
    import json
    json_path = os.path.join(args.output, "exhibits.json")
    exhibits_copy = [{k: v for k, v in ex.items() if k != 'mask'} for ex in exhibits]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'image_path': args.image,
            'model': args.model,
            'num_exhibits': len(exhibits),
            'exhibits': exhibits_copy
        }, f, indent=2, ensure_ascii=False)
    print(f"[+] 结果已保存到: {json_path}")

    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()
