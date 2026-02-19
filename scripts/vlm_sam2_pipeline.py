#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM + SAM2 联合分割流程

1. VLM (Qwen-VL) 识别展品及位置
2. SAM2 基于位置进行精细分割
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import subprocess

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 确保 sam2 可导入
def ensure_sam2_installed():
    """确保 sam2 包已安装"""
    try:
        from sam2.build_sam import build_sam2
        return True
    except ImportError:
        print("[*] SAM2 未安装，正在从 GitHub 安装...")
        try:
            # 安装官方 SAM2 包
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "git+https://github.com/facebookresearch/segment-anything-2.git"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("[+] SAM2 安装成功!")
                print("[*] 请重新运行脚本")
                return False
            else:
                print(f"[!] SAM2 安装失败: {result.stderr}")
                return False
        except Exception as e:
            print(f"[!] SAM2 安装异常: {e}")
            return False

# 检查并尝试安装
if not ensure_sam2_installed():
    print("\n请重新运行:")
    print("  python scripts/vlm_sam2_pipeline.py --image data/R.jpg")
    sys.exit(1)

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 导入配置
from config import Config


def encode_image_to_base64(image_path):
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def call_qwen_vlm(image_path, api_key=None):
    """
    使用 Qwen-VL 识别图片中的展品及位置

    返回格式:
    [
        {"name": "展品名称", "bbox": [x1, y1, x2, y2], "description": "描述"},
        ...
    ]
    """
    print("\n" + "=" * 60)
    print("[Step 1/2] VLM 识别 - 检测展品位置")
    print("=" * 60)

    config = Config()

    # 构建请求
    image_base64 = encode_image_to_base64(image_path)

    prompt = """请分析这张展厅图片，识别出所有值得观看的展品。

对于每个展品，请提供：
1. 展品类型（如：画作、雕塑、装置艺术等）
2. 在图片中的位置（边界框坐标 [x1, y1, x2, y2]，其中 (0,0) 是左上角）
3. 简短描述

请以 JSON 格式返回，格式如下：
[
  {
    "name": "展品名称",
    "type": "类型",
    "bbox": [x1, y1, x2, y2],
    "description": "描述"
  }
]

注意：
- 只识别真正的展品，忽略墙壁、地板、灯光等
- 边界框要紧凑地包围展品
- 坐标范围：x: 0-1100, y: 0-600"""

    print("[*] 调用 Qwen-VL API...")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key or config.qwen_api_key,
            base_url=config.qwen_base_url
        )

        response = client.chat.completions.create(
            model=config.vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content
        print(f"[+] VLM 响应:\n{result_text[:500]}...")

        # 解析 JSON
        import re
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            exhibits = json.loads(json_match.group())

            # 验证边界框
            valid_exhibits = []
            for ex in exhibits:
                bbox = ex.get('bbox', [])
                if len(bbox) == 4:
                    # 确保坐标在合理范围内
                    x1, y1, x2, y2 = bbox
                    if 0 <= x1 < x2 <= 1100 and 0 <= y1 < y2 <= 600:
                        valid_exhibits.append({
                            "name": ex.get('name', 'Unknown'),
                            "type": ex.get('type', 'Unknown'),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "description": ex.get('description', '')
                        })

            print(f"[+] 识别到 {len(valid_exhibits)} 个有效展品")
            return valid_exhibits

        else:
            print("[!] 无法解析 VLM 响应，使用默认检测")
            return []

    except Exception as e:
        print(f"[!] VLM 调用失败: {e}")
        return []


def sam2_refine_segmentation(image_path, vlm_exhibits, output_dir, model_path=None, config_path=None):
    """
    使用 SAM2 对 VLM 识别的区域进行精细分割

    Args:
        image_path: 图片路径
        vlm_exhibits: VLM 识别的展品列表
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("[Step 2/2] SAM2 精细分割")
    print("=" * 60)

    if model_path is None:
        model_path = os.path.join(project_root, "models/sam2/sam2_hiera_small.pt")
    if config_path is None:
        config_path = "configs/sam2/sam2_hiera_s.yaml"

    # 加载 SAM2 预测器
    print("[*] 加载 SAM2 模型...")
    sam2_model = build_sam2(
        config_file=config_path,
        ckpt_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    predictor = SAM2ImagePredictor(sam2_model)
    print("[+] 模型加载完成")

    # 加载图片
    image = np.array(Image.open(image_path))
    predictor.set_image(image)

    # 对每个 VLM 识别的区域进行精细分割
    refined_exhibits = []

    for i, exhibit in enumerate(vlm_exhibits):
        print(f"\n[*] 处理展品 #{i+1}: {exhibit['name']}")

        bbox = exhibit['bbox']
        x1, y1, x2, y2 = bbox

        # 使用边界框作为提示
        box = np.array([x1, y1, x2, y2])

        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=True,
        )

        # 选择最佳掩码
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]

        print(f"    置信度: {best_score:.3f}")

        # 计算精确的边界框和中心点
        rows = np.any(best_mask, axis=1)
        cols = np.any(best_mask, axis=0)

        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            refined_bbox = (int(cmin), int(rmin), int(cmax), int(rmax))
            center = (int((cmin + cmax) / 2), int((rmin + rmax) / 2))
            area = int((cmax - cmin) * (rmax - rmin))

            refined_exhibits.append({
                "id": f"EX-{i:03d}",
                "name": exhibit['name'],
                "type": exhibit['type'],
                "description": exhibit['description'],
                "original_bbox": bbox,  # VLM 的原始边界框
                "refined_bbox": refined_bbox,  # SAM2 精细分割的边界框
                "center": center,
                "area": area,
                "confidence": float(best_score)
            })

    print(f"\n[+] 完成！精炼分割了 {len(refined_exhibits)} 个展品")

    # 保存结果
    save_results(image_path, refined_exhibits, output_dir)

    return refined_exhibits


def save_results(image_path, exhibits, output_dir):
    """保存结果并绘制可视化"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON
    result = {
        "image_path": image_path,
        "num_exhibits": len(exhibits),
        "exhibits": exhibits
    }

    json_path = os.path.join(output_dir, "vlm_sam2_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[*] 保存: {json_path}")

    # 绘制可视化
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0)
    ]

    for i, ex in enumerate(exhibits):
        color = colors[i % len(colors)]

        # VLM 原始边界框（虚线效果）
        orig_bbox = ex['original_bbox']
        draw.rectangle(orig_bbox, outline=color, width=1)

        # SAM2 精细边界框（实线）
        ref_bbox = ex['refined_bbox']
        draw.rectangle(ref_bbox, outline=color, width=3)

        # 中心点
        center = ex['center']
        draw.ellipse([center[0]-5, center[1]-5, center[0]+5, center[1]+5],
                     fill=color, outline='white')

        # 标签
        label = f"{i+1}. {ex['name']}"
        draw.text((ref_bbox[0], ref_bbox[1] - 20), label, fill=color)

    # 保存图片
    overlay_path = os.path.join(output_dir, "vlm_sam2_overlay.png")
    img.save(overlay_path)
    print(f"[*] 保存: {overlay_path}")


def main():
    parser = argparse.ArgumentParser(description="VLM + SAM2 联合分割")
    parser.add_argument("--image", type=str, default="data/R.jpg", help="输入图片")
    parser.add_argument("--output", type=str, default="data/outputs/vlm_sam2", help="输出目录")
    parser.add_argument("--api-key", type=str, default=None, help="Qwen API Key")
    parser.add_argument("--model", type=str, default=None, help="SAM2 模型路径")
    parser.add_argument("--config", type=str, default=None, help="SAM2 配置路径")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] 错误: 图片不存在: {args.image}")
        return

    print("=" * 60)
    print("VLM + SAM2 联合分割流程")
    print("=" * 60)
    print(f"输入: {args.image}")
    print(f"输出: {args.output}")

    try:
        # Step 1: VLM 识别
        vlm_exhibits = call_qwen_vlm(args.image, args.api_key)

        if not vlm_exhibits:
            print("[!] VLM 未能识别展品，请检查 API 配置")
            return

        # Step 2: SAM2 精细分割
        refined_exhibits = sam2_refine_segmentation(
            args.image, vlm_exhibits, args.output,
            args.model, args.config
        )

        print(f"\n[OK] 完成！")
        print(f"    VLM 识别: {len(vlm_exhibits)} 个展品")
        print(f"    SAM2 精炼: {len(refined_exhibits)} 个展品")
        print(f"    输出目录: {args.output}/")

    except Exception as e:
        print(f"[!] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
