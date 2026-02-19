#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 2 调用脚本 - 用于图像分割

使用方法:
    python call_sam2.py --image data/R.jpg --output data/outputs/masks/
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加 sam2 模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam2'))


def call_sam2_online(image_path: str, api_token: str = None) -> dict:
    """
    使用在线 Replicate API 调用 SAM 2

    适合: 没有本地 GPU，或不想部署本地模型
    """
    import replicate
    from PIL import Image
    import numpy as np

    print(f"使用在线 API (Replicate) 调用 SAM 2...")
    print(f"  图片: {image_path}")

    # 设置 API token
    if api_token:
        os.environ['REPLICATE_API_TOKEN'] = api_token

    client = replicate.Client(api_token=api_token)

    # 调用 API
    with open(image_path, "rb") as f:
        output = client.run(
            "lucataco/segment-anything-2",
            input={"image": f}
        )

    # 解析输出
    return parse_replicate_output(output, image_path)


def call_sam2_local(image_path: str, model_path: str = "models/sam2/sam2-hiera-tiny.pt", device: str = "cuda") -> dict:
    """
    使用本地 SAM 2 模型进行分割

    适合: 有 NVIDIA GPU，已安装 SAM 2
    """
    import torch
    import numpy as np
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print(f"使用本地模型调用 SAM 2...")
    print(f"  图片: {image_path}")
    print(f"  模型: {model_path}")
    print(f"  设备: {device}")

    # 检查 CUDA
    if device == "cuda" and not torch.cuda.is_available():
        print("  ⚠️  CUDA 不可用，切换到 CPU")
        device = "cpu"

    # 构建模型
    try:
        sam2_model = build_sam2(
            config_file="sam2/configs/sam2-hiera-tiny.yaml",
            ckpt_path=model_path,
            device=device,
        )
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        print("\n提示: 请先运行以下命令安装 SAM 2:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        print("  wget https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2-hiera-tiny.pt -P models/sam2/")
        print("  git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo")
        print("  cp -r sam2_repo/sam2 .")
        return {"error": "模型加载失败"}

    # 创建预测器
    predictor = SAM2ImagePredictor(sam2_model)
    print("  ✓ 预测器创建成功")

    # 加载图片
    image = np.array(Image.open(image_path))
    print(f"  图片尺寸: {image.shape}")

    # 设置图片
    predictor.set_image(image)

    # 自动分割
    print("  运行分割...")
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=None,
        multimask_output=True,
    )

    print(f"  ✓ 分割完成! 检测到 {len(masks)} 个掩码")

    # 提取边界框和中心点
    boxes, centers = extract_boxes_and_centers(masks)

    return {
        "masks": masks,
        "scores": scores.tolist(),
        "boxes": boxes,
        "centers": centers,
        "image_shape": image.shape
    }


def parse_replicate_output(output: any, image_path: str) -> dict:
    """解析 Replicate API 输出"""
    # Replicate API 的输出格式需要根据实际情况调整
    # 这里提供基本框架
    return {
        "masks": [],
        "scores": [],
        "boxes": [],
        "centers": [],
        "image_path": image_path
    }


def extract_boxes_and_centers(masks):
    """从掩码提取边界框和中心点"""
    import numpy as np

    boxes = []
    centers = []

    for mask in masks:
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


def save_results(result: dict, output_dir: str):
    """保存分割结果"""
    import os
    from PIL import Image
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # 保存掩码
    for i, mask in enumerate(result.get('masks', [])):
        if isinstance(mask, list):
            mask_array = np.array(mask, dtype=np.uint8) * 255
        else:
            mask_array = (mask * 255).astype(np.uint8)

        mask_img = Image.fromarray(mask_array, mode='L')
        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        mask_img.save(mask_path)
        print(f"  保存掩码: {mask_path}")

    # 保存边界框信息
    boxes = result.get('boxes', [])
    centers = result.get('centers', [])
    scores = result.get('scores', [])

    info_path = os.path.join(output_dir, "segmentation_info.json")
    info = {
        "image_path": result.get("image_path"),
        "image_shape": result.get("image_shape"),
        "num_masks": len(boxes),
        "masks": [
            {
                "box": boxes[i] if i < len(boxes) else None,
                "center": centers[i] if i < len(centers) else None,
                "score": scores[i] if i < len(scores) else None
            }
            for i in range(len(boxes))
        ]
    }

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"  保存信息: {info_path}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="SAM 2 图像分割调用脚本")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, default="data/outputs/masks/", help="输出目录")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "local", "online"],
                       help="调用模式: auto(自动选择), local(本地模型), online(在线API)")
    parser.add_argument("--model", type=str, default="models/sam2/sam2-hiera-tiny.pt",
                       help="本地模型路径 (仅用于 local 模式)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="设备选择 (仅用于 local 模式)")

    args = parser.parse_args()

    # 检查图片是否存在
    if not os.path.exists(args.image):
        print(f"❌ 错误: 图片不存在: {args.image}")
        return

    # 调用 SAM 2
    if args.mode == "online":
        # 使用在线 API
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            print("❌ 错误: 未设置 REPLICATE_API_TOKEN 环境变量")
            print("请先设置: export REPLICATE_API_TOKEN=你的token")
            return
        result = call_sam2_online(args.image, api_token)
    elif args.mode == "local":
        # 使用本地模型
        result = call_sam2_local(args.image, args.model, args.device)
    else:  # auto
        # 自动检测
        model_path = args.model
        if os.path.exists(model_path):
            print(f"检测到本地模型: {model_path}")
            result = call_sam2_local(args.image, model_path, args.device)
        else:
            print("未检测到本地模型，使用在线 API...")
            api_token = os.environ.get("REPLICATE_API_TOKEN")
            if not api_token:
                print("❌ 错误: 未设置 REPLICATE_API_TOKEN 环境变量")
                return
            result = call_sam2_online(args.image, api_token)

    # 检查结果
    if "error" in result:
        print(f"❌ {result['error']}")
        return

    # 保存结果
    save_results(result, args.output)
    print(f"\n✅ 完成! 结果保存在: {args.output}")


if __name__ == "__main__":
    main()
