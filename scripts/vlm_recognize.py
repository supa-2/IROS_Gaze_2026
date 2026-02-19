#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 展品识别脚本

使用 Qwen-VL 识别图片中的展品及位置
"""

import os
import sys
import json
import argparse
import base64
from PIL import Image, ImageDraw

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def call_qwen_vlm(image_path):
    """使用 Qwen-VL 识别图片中的展品及位置"""
    print("=" * 60)
    print("VLM 展品识别")
    print("=" * 60)
    print(f"图片: {image_path}")

    # 获取 API 配置 - 优先使用 Qwen
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[!] 错误: 未找到 API Key")
        print("    请设置 QWEN_API_KEY 环境变量")
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = "qwen-vl-max-latest"

    print(f"API Key: {api_key[:20]}..." if len(api_key) > 20 else f"API Key: {api_key}")

    print(f"API: {base_url}")
    print(f"Model: {model}")

    # 编码图片
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    prompt = """请分析这张展厅图片，识别出所有值得观看的展品。

对于每个展品，请提供：
1. 展品名称（如：画作1、雕塑A）
2. 展品类型（如：画作、雕塑、装置艺术）
3. 在图片中的位置（边界框坐标 [x1, y1, x2, y2]，其中 (0,0) 是左上角，(1100, 600) 是右下角）
4. 简短描述

请以 JSON 格式返回：
[
  {
    "name": "展品名称",
    "type": "展品类型",
    "bbox": [x1, y1, x2, y2],
    "description": "简短描述"
  }
]

要求：
- 只识别真正的展品（画作、雕塑等），忽略墙壁、地板、展柜、灯光
- 边界框要紧凑地包围展品主体
- 坐标范围：x: 0-1100, y: 0-600
- 返回 3-10 个主要展品即可"""

    print("\n[*] 调用 Qwen-VL API...")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content
        print(f"\n[+] VLM 响应:")
        print(result_text[:500] + "..." if len(result_text) > 500 else result_text)

        # 解析 JSON
        import re
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            exhibits = json.loads(json_match.group())

            # 验证并过滤
            valid_exhibits = []
            for ex in exhibits:
                bbox = ex.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # 验证坐标范围
                    try:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if 0 <= x1 < x2 <= 1100 and 0 <= y1 < y2 <= 600:
                            # 验证面积合理（不要太小或太大）
                            area = (x2 - x1) * (y2 - y1)
                            if 1000 < area < 500000:  # 至少 30x30，最多 500x500
                                valid_exhibits.append({
                                    "name": ex.get('name', 'Unknown'),
                                    "type": ex.get('type', 'Unknown'),
                                    "bbox": [x1, y1, x2, y2],
                                    "description": ex.get('description', '')
                                })
                    except ValueError:
                        continue

            print(f"\n[+] 识别到 {len(valid_exhibits)} 个有效展品")

            return valid_exhibits if valid_exhibits else None

        else:
            print("[!] 无法解析 VLM 响应为 JSON")
            return None

    except Exception as e:
        print(f"[!] API 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(image_path, exhibits, output_dir):
    """保存结果并绘制可视化"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON
    result = {
        "image_path": image_path,
        "num_exhibits": len(exhibits),
        "exhibits": exhibits
    }

    json_path = os.path.join(output_dir, "vlm_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 保存: {json_path}")

    # 绘制可视化
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0),
        (128, 0, 255), (0, 128, 128), (128, 128, 0)
    ]

    for i, ex in enumerate(exhibits):
        color = colors[i % len(colors)]
        bbox = ex['bbox']
        x1, y1, x2, y2 = bbox

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        draw.ellipse([center_x-4, center_y-4, center_x+4, center_y+4],
                     fill=color, outline='white')

        # 绘制标签
        label = f"{i+1}. {ex['name']}"
        draw.text((x1, y1 - 15), label, fill=color)

    # 保存图片
    output_path = os.path.join(output_dir, "vlm_overlay.png")
    img.save(output_path)
    print(f"[*] 保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VLM 展品识别")
    parser.add_argument("--image", type=str, default="data/R.jpg", help="输入图片")
    parser.add_argument("--output", type=str, default="data/outputs/vlm", help="输出目录")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] 错误: 图片不存在: {args.image}")
        print("\n用法:")
        print("  python scripts/vlm_recognize.py --image data/R.jpg --output data/outputs/vlm")
        return

    print("\n请确认:")
    print(f"  图片存在: {os.path.exists(args.image)}")
    print(f"  QWEN_API_KEY 已设置: {'Yes' if os.getenv('QWEN_API_KEY') else 'No'}")

    if not os.getenv('QWEN_API_KEY'):
        print("\n[!] 请先设置 QWEN_API_KEY:")
        print("  export QWEN_API_KEY=你的Qwen_API密钥")
        return

    # 运行识别
    exhibits = call_qwen_vlm(args.image)

    if exhibits:
        save_results(args.image, exhibits, args.output)
        print(f"\n[OK] 完成! 识别到 {len(exhibits)} 个展品")
    else:
        print("\n[!] 识别失败")


if __name__ == "__main__":
    main()
