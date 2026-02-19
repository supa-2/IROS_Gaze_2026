#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 2 Small 模型调用脚本
适用于已部署 sam2-hiera-small 的服务器
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加 sam2 模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam2'))

import torch
import numpy as np
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_sam2_small(model_path: str = "models/sam2/sam2-hiera-small.pt",
                    config_path: str = "sam2/configs/sam2-hiera-small.yaml",
                    device: str = "auto") -> SAM2ImagePredictor:
    """
    加载 SAM 2 Small 模型

    Args:
        model_path: 模型文件路径
        config_path: 配置文件路径
        device: 设备选择 ("auto", "cuda", "cpu")

    Returns:
        SAM2ImagePredictor 预测器
    """
    print("=" * 60)
    print("SAM 2 Small 模型加载")
    print("=" * 60)

    # 自动检测设备
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[设备] CUDA - {torch.cuda.get_device_name(0)}")
            print(f"[显存] {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            print("[设备] CPU")

    # 检查文件存在性
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    print(f"[模型] {model_path}")
    print(f"[配置] {config_path}")

    # 加载模型
    print("\n正在加载模型...")
    sam2_model = build_sam2(
        config_file=config_path,
        ckpt_path=model_path,
        device=device,
    )

    print("✓ 模型加载成功!")

    # 创建预测器
    predictor = SAM2ImagePredictor(sam2_model)
    print("✓ 预测器创建成功!")

    return predictor


def segment_image(predictor: SAM2ImagePredictor, image_path: str,
                  auto_segment: bool = True) -> dict:
    """
    对图片进行分割

    Args:
        predictor: SAM2 预测器
        image_path: 图片路径
        auto_segment: 是否自动分割

    Returns:
        分割结果字典
    """
    print(f"\n[分割] 图片: {image_path}")

    # 加载图片
    image = np.array(Image.open(image_path))
    print(f"[尺寸] {image.shape}")

    # 设置图片
    predictor.set_image(image)
    print("✓ 图片已设置")

    if auto_segment:
        # 自动分割
        print("[模式] 自动分割")
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=True,
        )
    else:
        # 手动指定点分割
        print("[模式] 手动点分割")
        # 示例：在图片中心指定一个点
        h, w = image.shape[:2]
        point_coords = np.array([[w//2, h//2]])
        point_labels = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    print(f"✓ 分割完成! 检测到 {len(masks)} 个掩码")
    print(f"   置信度范围: {scores.min():.3f} ~ {scores.max():.3f}")

    # 提取边界框和中心点
    boxes, centers = extract_boxes_and_centers(masks)

    return {
        "masks": masks,
        "scores": scores.tolist(),
        "logits": logits.tolist(),
        "boxes": boxes,
        "centers": centers,
        "image_shape": image.shape,
        "image_path": image_path
    }


def extract_boxes_and_centers(masks):
    """从掩码提取边界框和中心点"""
    boxes = []
    centers = []

    for i, mask in enumerate(masks):
        if isinstance(mask, list):
            mask_array = np.array(mask)
        else:
            mask_array = mask

        # 计算边界框
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)

        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            boxes.append((int(cmin), int(rmin), int(cmax), int(rmax)))

            # 计算中心点
            center_x = int(np.mean([cmin, cmax]))
            center_y = int(np.mean([rmin, rmax]))
            centers.append((center_x, center_y))
        else:
            boxes.append((0, 0, 0, 0))
            centers.append((0, 0))

    return boxes, centers


def save_results(result: dict, output_dir: str, show_overlay: bool = True):
    """保存分割结果"""
    import os
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    # 保存掩码图片
    for i, mask in enumerate(result.get('masks', [])):
        if isinstance(mask, list):
            mask_array = np.array(mask, dtype=np.uint8) * 255
        else:
            mask_array = (mask * 255).astype(np.uint8)

        mask_img = Image.fromarray(mask_array, mode='L')
        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        mask_img.save(mask_path)
        print(f"  ✓ 掩码 {i}: {mask_path}")

    # 保存分割信息
    info = {
        "image_path": result.get("image_path"),
        "image_shape": result.get("image_shape"),
        "num_masks": len(result.get('masks', [])),
        "detections": []
    }

    for i, (box, center, score) in enumerate(zip(
        result.get('boxes', []),
        result.get('centers', []),
        result.get('scores', [])
    ):
        info["detections"].append({
            "id": i,
            "bbox": box,
            "center": center,
            "confidence": float(score)
        })

    info_path = os.path.join(output_dir, "segmentation_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 信息: {info_path}")

    # 创建叠加可视化
    if show_overlay and result.get('boxes'):
        create_overlay(result, output_dir)

    return output_dir


def create_overlay(result: dict, output_dir: str):
    """创建分割结果叠加图"""
    from PIL import Image

    image_path = result.get("image_path")
    if not os.path.exists(image_path):
        print("  ⚠️ 原图不存在，跳过叠加图生成")
        return

    image = Image.open(image_path).convert("RGBA")
    masks = result.get('masks', [])

    # 为每个掩码分配颜色
    colors = [
        (255, 0, 0, 128),    # 红色
        (0, 255, 0, 128),    # 绿色
        (0, 0, 255, 128),    # 蓝色
        (255, 255, 0, 128), # 黄色
        (255, 0, 255, 128), # 紫色
    ]

    for i, mask in enumerate(masks[:len(colors)]):
        color = colors[i % len(colors)]
        if isinstance(mask, list):
            mask_array = np.array(mask)
        else:
            mask_array = mask

        # 创建 RGBA 掩码
        mask_rgba = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        mask_rgba[mask_array] = color

        mask_img = Image.fromarray(mask_rgba, 'RGBA')

        # 叠加到原图上
        image = Image.alpha_composite(image.convert('RGBA'), mask_img)

    # 保存叠加图
    overlay_path = os.path.join(output_dir, "overlay.png")
    image.convert('RGB').save(overlay_path)
    print(f"  ✓ 叠加图: {overlay_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM 2 Small 模型调用")
    parser.add_argument("--image", type=str, help="输入图片路径")
    parser.add_argument("--output", type=str, default="data/outputs/sam2_small", help="输出目录")
    parser.add_argument("--model", type=str, default="models/sam2/sam2-hiera-small.pt", help="模型路径")
    parser.add_argument("--config", type=str, default="sam2/configs/sam2-hiera-small.yaml", help="配置文件路径")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="设备选择")
    parser.add_argument("--manual", action="store_true", help="使用手动点分割模式")

    args = parser.parse_args()

    # 检查图片
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ 错误: 图片不存在: {args.image}")
            return
        image_path = args.image
    else:
        # 使用默认测试图片
        image_path = "data/R.jpg"
        if not os.path.exists(image_path):
            print(f"❌ 默认图片不存在: {image_path}")
            print("请使用 --image 参数指定图片路径")
            return

    # 加载模型
    try:
        predictor = load_sam2_small(args.model, args.config, args.device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n请确认以下文件存在:")
        print(f"  - 模型: {args.model}")
        print(f"  - 配置: {args.config}")
        print("\n如果缺少配置文件，运行:")
        print("  cd ~ && git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo")
        print("  cp -r sam2_repo/sam2/configs ./sam2/")
        return

    # 执行分割
    result = segment_image(predictor, image_path, auto_segment=not args.manual)

    # 保存结果
    print(f"\n[保存结果]")
    save_results(result, args.output)

    print(f"\n✅ 完成! 结果保存在: {args.output}/")


if __name__ == "__main__":
    main()
