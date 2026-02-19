#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动可视化完整工作流

VLM 展品识别 + 眼动数据 + 热力图 + 轨迹图

使用方式:
    python scripts/full_pipeline.py --image data/R.jpg

    如果已有 VLM 识别结果:
    python scripts/full_pipeline.py --image data/R.jpg --vlm-result data/outputs/vlm/vlm_result.json
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from skills.visualization.pixel_heatmap import GazeHeatmapVisualizer, GazeRegion
from skills.visualization.gaze_trajectory import GazeTrajectoryVisualizer


# ============================================
# VLM 展品识别
# ============================================

def run_vlm_recognition(image_path, api_key=None):
    """使用 VLM 识别展品"""
    import base64
    import re

    print("\n" + "=" * 60)
    print("[Step 1/5] VLM 展品识别")
    print("=" * 60)

    # 获取 API key
    api_key_to_use = api_key or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key_to_use or api_key_to_use == "your_api_key_here":
        print("[!] 错误: 未找到有效的 API Key")
        print("    请在 .env 文件中设置 QWEN_API_KEY")
        return None

    base_url = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL",
                                                         "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("VLM_MODEL", "qwen-vl-max-latest")

    print(f"[*] API: {base_url}")
    print(f"[*] Model: {model}")

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

    print("[*] 调用 VLM API...")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key_to_use,
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
        print(f"[+] VLM 响应收到")

        # 解析 JSON
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            exhibits = json.loads(json_match.group())

            # 验证并过滤
            valid_exhibits = []
            for i, ex in enumerate(exhibits):
                bbox = ex.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    try:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if 0 <= x1 < x2 <= 1100 and 0 <= y1 < y2 <= 600:
                            area = (x2 - x1) * (y2 - y1)
                            if 1000 < area < 500000:
                                valid_exhibits.append({
                                    "id": f"EX-{i:03d}",
                                    "name": ex.get('name', f'展品{i+1}'),
                                    "type": ex.get('type', 'Unknown'),
                                    "bbox": [x1, y1, x2, y2],
                                    "description": ex.get('description', '')
                                })
                    except ValueError:
                        continue

            print(f"[+] 识别到 {len(valid_exhibits)} 个有效展品")
            return valid_exhibits

        else:
            print("[!] 无法解析 VLM 响应为 JSON")
            return None

    except Exception as e:
        print(f"[!] VLM 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# 绘制 VLM 识别结果
# ============================================

def draw_vlm_results(image_path, exhibits, output_path):
    """绘制 VLM 识别结果"""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 128), (128, 128, 0)
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

    img.save(output_path)
    print(f"[*] 保存: {output_path}")


# ============================================
# 生成模拟眼动数据
# ============================================

def generate_gaze_data(exhibits, num_records=25):
    """
    生成模拟眼动数据用于演示

    在实际使用中，应该使用真实的眼动追踪数据
    """
    print("\n" + "=" * 60)
    print("[Step 2/5] 生成模拟眼动数据")
    print("=" * 60)

    import random

    gaze_records = []
    attention_levels = ['A', 'B', 'C', 'D', 'E']
    durations = {'A': 120, 'B': 60, 'C': 30, 'D': 15, 'E': 5}

    # 模拟真实观看顺序：有些展品看多次，有些只看一次
    visit_pattern = []
    num_exhibits = len(exhibits)

    # 添加一些重复观看（模拟回看）
    for i in range(num_records):
        if i < num_exhibits:
            visit_pattern.append(i)
        else:
            # 随机选择之前的展品回看
            visit_pattern.append(random.randint(0, min(i, num_exhibits - 1)))

    for i, exhibit_idx in enumerate(visit_pattern):
        exhibit = exhibits[exhibit_idx]
        bbox = exhibit['bbox']
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        # 添加随机偏移
        x = center_x + random.randint(-25, 25)
        y = center_y + random.randint(-25, 25)

        level = random.choice(attention_levels)
        duration = durations[level] * random.uniform(0.5, 1.5)

        gaze_records.append({
            'x': x,
            'y': y,
            'duration': duration,
            'exhibit_id': exhibit['id'],
            'exhibit_name': exhibit['name'],
            'attention_level': level,
            'timestamp': datetime.now().isoformat(),
            'sequence': i + 1
        })

    print(f"[*] 生成了 {len(gaze_records)} 条眼动记录")

    return gaze_records


# ============================================
# 热力图可视化
# ============================================

def create_heatmap_visualization(image_path, exhibits, gaze_records, output_dir):
    """创建热力图可视化"""
    print("\n" + "=" * 60)
    print("[Step 3/5] 生成热力图")
    print("=" * 60)

    viz = GazeHeatmapVisualizer(image_path, sigma=50)

    # 添加展品区域
    for ex in exhibits:
        region = GazeRegion(
            id=ex['id'],
            label=ex['name'],
            type=ex['type'],
            bbox=tuple(ex['bbox'])
        )
        viz.regions[region.id] = region

    # 添加眼动记录
    for record in gaze_records:
        viz.add_gaze_record(
            region_id=record['exhibit_id'],
            duration=record['duration'],
            x=record['x'],
            y=record['y']
        )

    # 计算热力图
    viz.calculate_heatmap()

    # 获取统计
    stats = viz.get_region_statistics()
    print(f"[*] 眼动统计: {len(stats)} 个区域有凝视数据")
    for rid, rstat in stats.items():
        print(f"    {rstat['label']}: {rstat['fixation_count']} 次凝视, {rstat['total_duration']:.1f}s")

    # 保存统计
    stats_path = os.path.join(output_dir, "heatmap_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 生成热力图
    overlay_path = os.path.join(output_dir, "heatmap_overlay.png")
    viz.visualize_overlay(overlay_path, alpha=0.6)

    comparison_path = os.path.join(output_dir, "heatmap_comparison.png")
    viz.visualize_side_by_side(comparison_path)

    return viz, stats


# ============================================
# 轨迹可视化
# ============================================

def create_trajectory_visualization(image_path, exhibits, gaze_records, heatmap_data, output_dir):
    """创建轨迹可视化"""
    print("\n" + "=" * 60)
    print("[Step 4/5] 生成轨迹图")
    print("=" * 60)

    traj_viz = GazeTrajectoryVisualizer(image_path)

    # 添加区域
    for ex in exhibits:
        traj_viz.add_region(
            region_id=ex['id'],
            label=ex['name'],
            bbox=ex['bbox']
        )

    # 添加轨迹点
    for record in gaze_records:
        traj_viz.add_trajectory_point(
            x=record['x'],
            y=record['y'],
            duration=record['duration'],
            region_id=record['exhibit_id']
        )

    # 生成轨迹图
    trajectory_path = os.path.join(output_dir, "gaze_trajectory.png")
    traj_viz.visualize(trajectory_path)

    # 生成轨迹+热力图叠加
    combined_path = os.path.join(output_dir, "trajectory_heatmap_combined.png")
    traj_viz.visualize_with_heatmap(
        heatmap_data,
        combined_path,
        heatmap_alpha=0.5
    )

    # 统计
    traj_stats = traj_viz.get_statistics()
    print(f"[*] 轨迹统计: {traj_stats['total_fixations']} 次凝视, "
          f"{traj_stats['unique_regions']} 个区域, "
          f"{traj_stats['region_transitions']} 次转换")

    # 保存统计
    traj_stats_path = os.path.join(output_dir, "trajectory_statistics.json")
    with open(traj_stats_path, 'w', encoding='utf-8') as f:
        json.dump(traj_stats, f, indent=2, ensure_ascii=False)

    return traj_viz, traj_stats


# ============================================
# 生成综合报告
# ============================================

def generate_final_report(exhibits, gaze_records, heatmap_stats, traj_stats, output_dir):
    """生成综合报告"""
    print("\n" + "=" * 60)
    print("[Step 5/5] 生成综合报告")
    print("=" * 60)

    # 计算总统计
    total_duration = sum(r['duration'] for r in gaze_records)
    avg_duration = total_duration / len(gaze_records) if gaze_records else 0

    level_counts = {}
    for r in gaze_records:
        level = r['attention_level']
        level_counts[level] = level_counts.get(level, 0) + 1

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_exhibits": len(exhibits),
            "total_gaze_records": len(gaze_records),
            "unique_exhibits_visited": len(set(r['exhibit_id'] for r in gaze_records)),
            "total_duration": round(total_duration, 1),
            "average_duration": round(avg_duration, 1),
            "attention_level_distribution": level_counts
        },
        "exhibits": exhibits,
        "gaze_records": gaze_records,
        "heatmap_statistics": heatmap_stats,
        "trajectory_statistics": traj_stats
    }

    # 保存报告
    report_path = os.path.join(output_dir, "FINAL_REPORT.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "=" * 60)
    print("【最终报告】")
    print("=" * 60)
    print(f"检测展品数: {report['summary']['total_exhibits']}")
    print(f"眼动记录数: {report['summary']['total_gaze_records']}")
    print(f"访问展品数: {report['summary']['unique_exhibits_visited']}")
    print(f"总观看时长: {report['summary']['total_duration']:.1f}秒")
    print(f"平均停留: {report['summary']['average_duration']:.1f}秒")
    print(f"注意力分布: {level_counts}")
    print(f"\n输出目录: {output_dir}/")
    print("=" * 60)

    return report


# ============================================
# Main Pipeline
# ============================================

def run_full_pipeline(image_path, vlm_result_path=None, output_dir=None, use_vlm=True):
    """
    运行完整可视化流程

    Args:
        image_path: 原始图片路径
        vlm_result_path: VLM识别结果JSON路径（可选）
        output_dir: 输出目录
        use_vlm: 是否运行VLM识别（False时使用已有结果）
    """
    print("\n" + "=" * 60)
    print("眼动可视化完整工作流")
    print("=" * 60)
    print(f"输入: {image_path}")

    # 确定输出目录
    if output_dir is None:
        output_dir = f"data/outputs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"输出: {output_dir}")

    # 步骤1: VLM展品识别（或加载已有结果）
    exhibits = None

    if vlm_result_path and os.path.exists(vlm_result_path):
        # 加载已有结果
        print(f"\n[*] 加载已有VLM识别结果: {vlm_result_path}")
        with open(vlm_result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        exhibits = data.get('exhibits', [])

        # 为没有id的展品添加id
        for i, ex in enumerate(exhibits):
            if 'id' not in ex:
                ex['id'] = f"EX-{i:03d}"

        print(f"[+] 加载了 {len(exhibits)} 个展品")

        # 复制识别图
        import shutil
        overlay_src = os.path.join(os.path.dirname(vlm_result_path), "vlm_overlay.png")
        if os.path.exists(overlay_src):
            shutil.copy(overlay_src, os.path.join(output_dir, "01_vlm_recognition.png"))

    elif use_vlm:
        # 运行VLM识别
        exhibits = run_vlm_recognition(image_path)

        if exhibits:
            # 保存VLM结果
            vlm_result = {
                "image_path": image_path,
                "num_exhibits": len(exhibits),
                "exhibits": exhibits
            }
            vlm_path = os.path.join(output_dir, "vlm_result.json")
            with open(vlm_path, 'w', encoding='utf-8') as f:
                json.dump(vlm_result, f, indent=2, ensure_ascii=False)

            # 绘制识别结果
            draw_vlm_results(image_path, exhibits, os.path.join(output_dir, "01_vlm_recognition.png"))

    if not exhibits:
        print("[!] 错误: 没有可用的展品数据")
        return None

    # 步骤2: 生成眼动数据
    gaze_records = generate_gaze_data(exhibits, num_records=25)

    # 保存眼动数据
    gaze_path = os.path.join(output_dir, "gaze_records.json")
    with open(gaze_path, 'w', encoding='utf-8') as f:
        json.dump(gaze_records, f, indent=2, ensure_ascii=False)

    # 步骤3: 热力图可视化
    heatmap_viz, heatmap_stats = create_heatmap_visualization(
        image_path, exhibits, gaze_records, output_dir
    )

    # 步骤4: 轨迹可视化
    traj_viz, traj_stats = create_trajectory_visualization(
        image_path, exhibits, gaze_records,
        heatmap_viz.heatmap_data, output_dir
    )

    # 步骤5: 综合报告
    report = generate_final_report(
        exhibits, gaze_records, heatmap_stats, traj_stats, output_dir
    )

    print("\n" + "=" * 60)
    print("[完成] 所有可视化已生成!")
    print(f"输出目录: {output_dir}/")
    print("=" * 60)

    return {
        'output_dir': output_dir,
        'exhibits': exhibits,
        'gaze_records': gaze_records,
        'report': report
    }


def main():
    parser = argparse.ArgumentParser(description="眼动可视化完整工作流")
    parser.add_argument("--image", type=str, default="data/R.jpg",
                       help="输入图片路径")
    parser.add_argument("--vlm-result", type=str, default=None,
                       help="VLM识别结果JSON路径（默认自动查找）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出目录（默认自动生成）")
    parser.add_argument("--no-vlm", action="store_true",
                       help="不运行VLM识别，使用已有结果")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] 错误: 图片不存在: {args.image}")
        return

    # 如果没有指定VLM结果路径，尝试自动查找
    vlm_result_path = args.vlm_result
    if vlm_result_path is None:
        possible_paths = [
            "data/outputs/vlm/vlm_result.json",
            "data/outputs/vlm_sam2/vlm_sam2_result.json"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                vlm_result_path = path
                print(f"[*] 自动找到VLM结果: {path}")
                break

    run_full_pipeline(
        image_path=args.image,
        vlm_result_path=vlm_result_path,
        output_dir=args.output,
        use_vlm=not args.no_vlm and vlm_result_path is None
    )


if __name__ == "__main__":
    main()
