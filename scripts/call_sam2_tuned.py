#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 优化配置分割脚本
调整参数以获得更好的分割效果
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def load_sam2_tuned(model_path=None, config_path=None, device="auto", preset="balanced"):
    """
    加载 SAM2 模型，使用优化参数

    preset: 预设配置
        - "precise": 精确模式（更多检测，更慢）
        - "balanced": 平衡模式（推荐）
        - "fast": 快速模式（较少检测，更快）
    """
    if model_path is None:
        model_path = os.path.join(project_root, "models/sam2/sam2_hiera_small.pt")
    if config_path is None:
        config_path = "configs/sam2/sam2_hiera_s.yaml"

    print("=" * 60)
    print("SAM2 Tuned Configuration")
    print("=" * 60)
    print(f"Preset: {preset}")

    # 预设配置
    presets = {
        "precise": {
            "points_per_side": 128,          # 高密度采样
            "pred_iou_thresh": 0.5,         # 较低 IoU 阈值
            "stability_score_thresh": 0.7,   # 较低稳定性阈值
            "min_mask_region_area": 100,     # 保留更小区域
            "boxes_nms_thresh": 0.7,
        },
        "balanced": {
            "points_per_side": 64,
            "pred_iou_thresh": 0.6,
            "stability_score_thresh": 0.75,
            "min_mask_region_area": 500,
            "boxes_nms_thresh": 0.7,
        },
        "fast": {
            "points_per_side": 32,
            "pred_iou_thresh": 0.7,
            "stability_score_thresh": 0.85,
            "min_mask_region_area": 1000,
            "boxes_nms_thresh": 0.7,
        }
    }

    config = presets.get(preset, presets["balanced"])

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Device] {device}")

    print(f"[Model] {model_path}")
    print(f"[Config] {config_path}")
    print("\nParameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    sam2_model = build_sam2(
        config_file=config_path,
        ckpt_path=model_path,
        device=device,
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        **config
    )

    print("\n[+] Model loaded!")
    return mask_generator, config


def segment_and_save(mask_generator, image_path, output_dir):
    """分割并保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[*] Processing: {image_path}")
    image = np.array(Image.open(image_path))
    print(f"    Size: {image.shape}")

    print("[*] Running segmentation...")
    masks = mask_generator.generate(image)

    print(f"[+] Found {len(masks)} masks")

    # 过滤和排序
    valid_masks = [m for m in masks if m.get('predicted_iou', 0) > 0.5]
    valid_masks.sort(key=lambda x: x['area'], reverse=True)

    print(f"    Valid (IoU > 0.5): {len(valid_masks)}")

    # 提取展品信息
    exhibits = []
    for i, mask_data in enumerate(valid_masks[:20]):  # 最多20个
        bbox = mask_data.get('bbox', [0, 0, 0, 0])
        x1, y1, w, h = bbox

        exhibits.append({
            "id": f"EX-{i:03d}",
            "bbox": (int(x1), int(y1), int(x1 + w), int(y1 + h)),
            "center": (int(x1 + w / 2), int(y1 + h / 2)),
            "area": mask_data.get('area', 0),
            "confidence": mask_data.get('predicted_iou', 0.0)
        })

    # 保存结果
    info = {
        "image_path": image_path,
        "image_shape": list(image.shape),
        "total_masks": len(masks),
        "valid_masks": len(valid_masks),
        "exhibits": exhibits
    }

    info_path = os.path.join(output_dir, "segmentation_tuned.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # 绘制检测图
    draw_results(image, valid_masks[:15], os.path.join(output_dir, "overlay.png"))

    return exhibits


def draw_results(image, masks, output_path):
    """绘制检测结果"""
    from PIL import ImageDraw, ImageFont

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 128), (128, 128, 0), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 128)
    ]

    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i, mask_data in enumerate(masks):
        color = colors[i % len(colors)]
        bbox = mask_data.get('bbox', [0, 0, 0, 0])
        x, y, w, h = bbox
        conf = mask_data.get('predicted_iou', 0)
        area = mask_data.get('area', 0)

        # 绘制掩码
        mask = mask_data['segmentation']
        mask_img = Image.fromarray((mask * 100).astype(np.uint8), mode='L')
        mask_rgba = Image.new("RGBA", mask_img.size, color + (80,))
        img_rgba = img.convert("RGBA")
        img_rgba = Image.alpha_composite(img_rgba, mask_rgba)
        img = img_rgba.convert("RGB")

        # 绘制边界框
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        # 标签
        label = f"#{i+1} ({conf:.2f})"
        draw.text((x, y - 15), label, fill=color)

    img.save(output_path)
    print(f"[*] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM2 Tuned Segmentation")
    parser.add_argument("--image", type=str, default="data/R.jpg", help="Input image")
    parser.add_argument("--output", type=str, default="data/outputs/sam2_tuned", help="Output directory")
    parser.add_argument("--preset", type=str, default="balanced",
                       choices=["precise", "balanced", "fast"],
                       help="Configuration preset")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--config", type=str, default=None, help="Config path")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] Error: Image not found: {args.image}")
        return

    try:
        mask_generator, config = load_sam2_tuned(
            args.model, args.config, preset=args.preset
        )
        exhibits = segment_and_save(mask_generator, args.image, args.output)

        print(f"\n[Done] Found {len(exhibits)} exhibits")
        print(f"[Output] {args.output}/")

    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
