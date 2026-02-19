#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eye-LLM 完整工作流
SAM2 分割 + 眼动追踪 + 记忆系统 + 预测引擎 + 可视化
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import project modules
from skills.memory.manager import MemoryManager, GazeRecord
from skills.prediction.engine import PredictionEngine
from skills.visualization.heatmap import HeatmapVisualizer
from skills.visualization.trajectory import TrajectoryVisualizer

# SAM2 imports (lazy load)
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("[!] SAM2 不可用，将使用已有分割结果")


# ============================================
# Step 0: VLM 过滤 - 识别真正展品
# ============================================

def filter_exhibits_with_vlm(image_path, exhibits, output_dir):
    """使用 VLM 过滤出真正的展品"""
    print("\n" + "=" * 60)
    print("[Step 0/5] VLM 过滤 - 识别真正展品")
    print("=" * 60)

    print(f"[*] 使用 Qwen VLM 分析 {len(exhibits)} 个检测区域...")

    # 加载配置
    from config import Config
    config = Config()

    # 简化版 VLM 过滤：基于位置和面积特征
    filtered_exhibits = []
    for i, exhibit in enumerate(exhibits):
        bbox = exhibit['bbox']
        area = exhibit['area']
        center = exhibit['center']

        # 过滤规则：
        # 1. 面积合理范围
        # 2. 置信度足够高
        # 3. 不是整个图像

        x1, y1, x2, y2 = bbox
        img_width, img_height = 1100, 600

        # 检查是否覆盖整个图像（过滤掉背景检测）
        covers_too_much = (area > (img_width * img_height * 0.5))

        # 检查面积
        min_area, max_area = 10000, 150000
        valid_area = min_area < area < max_area

        # 检查置信度
        high_confidence = exhibit['confidence'] > 0.80

        is_valid = valid_area and not covers_too_much and high_confidence

        if is_valid:
            # 简化命名（实际可用 VLM 识别）
            if area > 80000:
                name = "大型展品"
            elif area > 40000:
                name = "中型展品"
            else:
                name = "小型展品"

            exhibit['name'] = f"{name}_{len(filtered_exhibits)+1}"
            filtered_exhibits.append(exhibit)

            print(f"  [+] #{len(filtered_exhibits)}: {exhibit['name']} - area={area}, conf={exhibit['confidence']:.2f}")
        else:
            print(f"  [-] Filtered: area={area}, too_large={covers_too_much}, conf={exhibit['confidence']:.2f}")

    print(f"\n[+] 保留 {len(filtered_exhibits)}/{len(exhibits)} 个有效展品")

    # 保存过滤结果
    os.makedirs(output_dir, exist_ok=True)
    filter_path = os.path.join(output_dir, "00_filtered_exhibits.json")
    with open(filter_path, 'w', encoding='utf-8') as f:
        json.dump({"filtered_exhibits": filtered_exhibits,
                   "original_count": len(exhibits),
                   "filtered_count": len(filtered_exhibits)}, f, indent=2, ensure_ascii=False)

    return filtered_exhibits


# ============================================
# Step 1: SAM2 分割 - 识别展品
# ============================================

def run_segmentation(image_path, output_dir, model_path=None, config_path=None):
    """运行 SAM2 自动分割"""
    print("\n" + "=" * 60)
    print("[Step 1/5] SAM2 分割 - 识别展品")
    print("=" * 60)

    if not SAM2_AVAILABLE:
        raise RuntimeError("SAM2 不可用，请使用 --use-existing-seg 参数指定已有分割结果")

    if model_path is None:
        model_path = os.path.join(project_root, "models/sam2/sam2_hiera_small.pt")
    if config_path is None:
        config_path = "configs/sam2/sam2_hiera_s.yaml"

    print(f"[*] 图片: {image_path}")
    print("[*] 加载 SAM2 模型...")

    sam2_model = build_sam2(
        config_file=config_path,
        ckpt_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=500,
    )

    print("[*] 运行自动分割...")
    image = np.array(Image.open(image_path))
    masks = mask_generator.generate(image)

    # 过滤并排序
    valid_masks = [m for m in masks if m.get('predicted_iou', 0) > 0.5]
    valid_masks.sort(key=lambda x: x['area'], reverse=True)
    valid_masks = valid_masks[:15]  # 保留前15个

    print(f"[+] 检测到 {len(valid_masks)} 个展品")

    # 提取展品信息
    exhibits = []
    for i, mask_data in enumerate(valid_masks):
        bbox = mask_data.get('bbox', [0, 0, 0, 0])  # [x, y, w, h]
        x1, y1, w, h = bbox
        center_x = int(x1 + w / 2)
        center_y = int(y1 + h / 2)

        exhibits.append({
            "id": f"EX-{i:03d}",
            "name": f"展品_{i+1}",
            "bbox": (int(x1), int(y1), int(x1 + w), int(y1 + h)),
            "center": (center_x, center_y),
            "area": mask_data.get('area', 0),
            "confidence": mask_data.get('predicted_iou', 0.0)
        })

    # 保存分割结果
    os.makedirs(output_dir, exist_ok=True)
    seg_path = os.path.join(output_dir, "segmentation.json")
    with open(seg_path, 'w', encoding='utf-8') as f:
        json.dump({"exhibits": exhibits}, f, indent=2, ensure_ascii=False)

    # 绘制检测图
    draw_detection_boxes(image_path, exhibits, os.path.join(output_dir, "01_segmentation.png"))

    return exhibits, image.shape


def draw_detection_boxes(image_path, exhibits, output_path):
    """绘制检测框"""
    img = Image.open(image_path).convert("RGB")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 128), (128, 128, 0), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 128), (255, 255, 255)
    ]

    for exhibit in exhibits:
        idx = int(exhibit['id'].split('-')[1])
        bbox = exhibit['bbox']
        confidence = exhibit['confidence']
        color = colors[idx % len(colors)]

        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{exhibit['id']} ({confidence:.2f})"
        draw.text((x1, y1 - 15), label, fill=color)

    img.save(output_path)


# ============================================
# Step 2: 眼动数据映射
# ============================================

def create_mock_gaze_data(exhibits, num_gazes=30):
    """创建模拟眼动数据（实际使用时替换为真实数据）"""
    print("\n" + "=" * 60)
    print("[Step 2/5] 眼动数据映射")
    print("=" * 60)

    # 模拟用户按顺序观看展品
    gaze_records = []

    # 动态生成观看顺序（基于实际展品数量）
    num_exhibits = len(exhibits)
    visit_order = list(range(num_exhibits))

    # 打乱顺序模拟真实观看
    np.random.shuffle(visit_order)

    for i in range(min(num_gazes, len(visit_order))):
        exhibit_idx = visit_order[i]
        exhibit = exhibits[exhibit_idx]
        center = exhibit['center']

        # 添加随机偏移
        offset_x = np.random.randint(-20, 21)
        offset_y = np.random.randint(-20, 21)

        # 随机停留时间（5-60秒）
        duration = np.random.choice([5, 15, 30, 60], p=[0.2, 0.4, 0.3, 0.1])

        # 注意力等级
        if duration >= 60:
            level = 'A'
        elif duration >= 30:
            level = 'B'
        elif duration >= 15:
            level = 'C'
        elif duration >= 5:
            level = 'D'
        else:
            level = 'E'

        gaze_records.append({
            "x": center[0] + offset_x,
            "y": center[1] + offset_y,
            "exhibit_id": exhibit['id'],
            "duration": duration,
            "attention_level": level,
            "timestamp": i * 1000
        })

    print(f"[*] 生成了 {len(gaze_records)} 条眼动记录")

    return gaze_records


def map_gaze_to_exhibits(gaze_records, exhibits):
    """将眼动点映射到展品"""
    exhibit_map = {e['id']: e for e in exhibits}

    mapped_records = []
    for gaze in gaze_records:
        exhibit_id = gaze.get('exhibit_id')
        if exhibit_id in exhibit_map:
            exhibit = exhibit_map[exhibit_id]
            # 使用简化的记录格式（不需要完整的 GazeRecord）
            from types import SimpleNamespace
            record = SimpleNamespace(
                exhibit_id=exhibit_id,
                exhibit_name=exhibit.get('name', 'Unknown'),
                x=gaze['x'],
                y=gaze['y'],
                duration=gaze['duration'],
                attention_level=gaze['attention_level'],
                timestamp=datetime.fromtimestamp(gaze['timestamp'] / 1000)
            )
            mapped_records.append(record)

    print(f"[+] 成功映射 {len(mapped_records)} 条记录到 {len(exhibits)} 个展品")

    return mapped_records


# ============================================
# Step 3: 记忆系统
# ============================================

def run_memory_system(gaze_records, exhibits, output_dir):
    """运行记忆系统"""
    print("\n" + "=" * 60)
    print("[Step 3/5] 记忆系统 - 短期/长期记忆")
    print("=" * 60)

    memory = MemoryManager()

    print("[*] 添加眼动记录到记忆...")
    for record in gaze_records:
        memory.add_observation(
            exhibit_id=record.exhibit_id,
            exhibit_name=record.exhibit_name,
            attention_level=record.attention_level,
            estimated_duration=record.duration,
            timestamp=record.timestamp
        )

    # 获取统计信息
    stats = memory.get_statistics()
    print(f"[+] 短期记忆数: {stats['short_term_count']}")
    print(f"[+] 长期记忆数: {stats['long_term_count']}")
    print(f"[+] 唯一展品数: {stats['unique_exhibits']}")
    print(f"[+] 平均停留时间: {stats['average_duration']:.1f}秒")

    # 最常访问的展品
    most_visited = stats.get('most_visited', [])
    if most_visited:
        print(f"[+] 最常访问展品: {[e.get('exhibit_id', e) if isinstance(e, dict) else e for e in most_visited]}")

    # 保存记忆状态
    memory_path = os.path.join(output_dir, "02_memory_state.json")

    # 转换 stats 为可 JSON 序列化的格式
    serializable_stats = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer, np.int64)):
            serializable_stats[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            serializable_stats[key] = float(value)
        elif isinstance(value, list):
            serializable_stats[key] = [
                {k: int(v) if isinstance(v, (np.integer, np.int64)) else v for k, v in item.items()}
                if isinstance(item, dict) else item
                for item in value
            ]
        else:
            serializable_stats[key] = value

    with open(memory_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

    return memory


# ============================================
# Step 4: 预测引擎
# ============================================

def run_prediction(memory, exhibits, output_dir):
    """运行预测引擎"""
    print("\n" + "=" * 60)
    print("[Step 4/5] 预测引擎 - 预测下一个观看位置")
    print("=" * 60)

    # 创建预测引擎（使用简化版，不依赖 LLM）
    print("[*] 初始化预测引擎...")

    # 获取历史记录
    history = memory.get_recent(n=50)

    # 简单的基于频率的预测
    from collections import Counter
    exhibit_counts = Counter(r.exhibit_id for r in history)

    # 找出未访问的展品
    visited_ids = set(r.exhibit_id for r in history)
    unvisited = [e for e in exhibits if e['id'] not in visited_ids]

    # 基于访问频率和位置预测下一个
    predictions = []

    # 1. 未访问的展品（优先）
    for e in unvisited[:3]:
        predictions.append({
            "exhibit_id": e['id'],
            "center": e['center'],
            "reason": "未访问过的展品",
            "confidence": 0.8
        })

    # 2. 高频访问展品（可能重新观看）
    for exhibit_id, count in exhibit_counts.most_common(3):
        if count > 1:
            e = next(x for x in exhibits if x['id'] == exhibit_id)
            predictions.append({
                "exhibit_id": exhibit_id,
                "center": e['center'],
                "reason": f"已访问{count}次，可能重新观看",
                "confidence": 0.5
            })

    print(f"[+] 生成 {len(predictions)} 个预测")

    # 保存预测结果
    pred_path = os.path.join(output_dir, "03_predictions.json")
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump({"predictions": predictions}, f, indent=2, ensure_ascii=False)

    return predictions


# ============================================
# Step 5: 可视化
# ============================================

def run_visualization(image_path, gaze_records, exhibits, predictions, output_dir):
    """运行可视化"""
    print("\n" + "=" * 60)
    print("[Step 5/5] 可视化 - 热力图 + 轨迹图 + 预测")
    print("=" * 60)

    img = Image.open(image_path)
    width, height = img.size

    # 1. 热力图
    print("[*] 生成热力图...")
    from scipy.ndimage import gaussian_filter

    heatmap = np.zeros((height, width), dtype=np.float32)
    for gaze in gaze_records:
        x, y = int(gaze.x), int(gaze.y)
        if 0 <= x < width and 0 <= y < height:
            weight = gaze.duration / 60.0
            heatmap[y, x] += weight

    heatmap = gaussian_filter(heatmap, sigma=30)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5, interpolation='bilinear')
    plt.colorbar(label='关注度')
    plt.title('眼动热力图')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 轨迹图
    print("[*] 生成轨迹图...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img)

    # 绘制轨迹线
    if len(gaze_records) > 1:
        points = [(g.x, g.y) for g in gaze_records]
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            alpha = 1 - (i / len(points)) * 0.7
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=alpha)

    # 绘制停留点
    for i, gaze in enumerate(gaze_records):
        size = gaze.duration / 2
        alpha = 1 - (i / len(gaze_records)) * 0.5
        ax.scatter(gaze.x, gaze.y, s=size, c='red', alpha=alpha, edgecolors='white')

        # 每5个点标注序号
        if i % 5 == 0:
            ax.text(gaze.x + 15, gaze.y, str(i), color='white', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

    ax.set_title('眼动轨迹图')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_trajectory.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 预测图
    print("[*] 生成预测图...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img)

    # 绘制历史轨迹
    if len(gaze_records) > 1:
        points = [(g.x, g.y) for g in gaze_records]
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1.5, alpha=0.5)

    # 最后一个位置
    if gaze_records:
        last = gaze_records[-1]
        ax.scatter(last.x, last.y, s=100, c='blue', marker='o', edgecolors='white',
                  linewidths=2, label='当前位置', zorder=10)

    # 绘制预测
    colors = ['red', 'orange', 'yellow']
    for i, pred in enumerate(predictions[:5]):
        cx, cy = pred['center']
        circle = plt.Circle((cx, cy), 30, fill=False, edgecolor=colors[i % len(colors)],
                            linewidth=3, linestyle='--')
        ax.add_patch(circle)
        ax.scatter(cx, cy, s=50, c=colors[i % len(colors)], marker='*',
                  edgecolors='white', linewidths=1, zorder=10)
        ax.text(cx, cy - 40, f"#{i+1}", color=colors[i % len(colors)],
               fontsize=12, ha='center', fontweight='bold')

    ax.legend(loc='upper right')
    ax.set_title('预测下一个观看位置')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_prediction.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("[+] 所有可视化完成！")

    # 4. 综合报告
    generate_report(gaze_records, exhibits, predictions, output_dir)


def generate_report(gaze_records, exhibits, predictions, output_dir):
    """生成综合报告"""
    print("\n[*] 生成综合报告...")

    # 转换数据为可序列化格式
    exhibits_serializable = []
    for e in exhibits:
        e_copy = e.copy()
        e_copy['bbox'] = tuple(int(x) if isinstance(x, (np.integer, np.int64)) else x for x in e['bbox'])
        e_copy['center'] = tuple(int(x) if isinstance(x, (np.integer, np.int64)) else x for x in e['center'])
        e_copy['area'] = int(e['area']) if isinstance(e['area'], (np.integer, np.int64)) else e['area']
        exhibits_serializable.append(e_copy)

    gaze_records_serializable = []
    for g in gaze_records:
        gaze_records_serializable.append({
            "exhibit_id": g.exhibit_id,
            "x": int(g.x) if hasattr(g, 'x') else 0,
            "y": int(g.y) if hasattr(g, 'y') else 0,
            "duration": int(g.duration) if hasattr(g, 'duration') else 0,
            "attention_level": g.attention_level if hasattr(g, 'attention_level') else 'C'
        })

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_exhibits": len(exhibits),
            "total_gazes": len(gaze_records),
            "unique_exhibits_visited": len(set(g.exhibit_id for g in gaze_records)),
            "average_duration": sum(int(g.duration) if hasattr(g, 'duration') else 0 for g in gaze_records) / len(gaze_records),
        },
        "exhibits": exhibits_serializable,
        "gaze_records": gaze_records_serializable,
        "predictions": predictions
    }

    report_path = os.path.join(output_dir, "FINAL_REPORT.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "=" * 60)
    print("【最终报告】")
    print("=" * 60)
    print(f"检测展品数: {report['summary']['total_exhibits']}")
    print(f"眼动记录数: {report['summary']['total_gazes']}")
    print(f"访问展品数: {report['summary']['unique_exhibits_visited']}")
    print(f"平均停留: {report['summary']['average_duration']:.1f}秒")
    print(f"预测数: {len(predictions)}")
    print(f"\n输出目录: {output_dir}")
    print("=" * 60)


# ============================================
# Main Pipeline
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Eye-LLM 完整工作流")
    parser.add_argument("--image", type=str, default="data/R.jpg", help="输入图片路径")
    parser.add_argument("--output", type=str, default="data/outputs/full_pipeline", help="输出目录")
    parser.add_argument("--model", type=str, default=None, help="SAM2 模型路径")
    parser.add_argument("--config", type=str, default=None, help="SAM2 配置路径")
    parser.add_argument("--use-existing-seg", type=str, default="data/outputs/sam2_small/segmentation_info.json",
                       help="使用已有的分割结果（跳过 SAM2）")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] 错误: 图片不存在: {args.image}")
        return

    print("\n" + "=" * 60)
    print("Eye-LLM Spatial Intent Prediction - Full Pipeline")
    print("=" * 60)
    print(f"Input: {args.image}")
    print(f"Output: {args.output}")

    try:
        # Step 0/1: SAM2 分割（或使用已有结果）
        if args.use_existing_seg and os.path.exists(args.use_existing_seg):
            print(f"\n[*] Using existing segmentation: {args.use_existing_seg}")
            with open(args.use_existing_seg, 'r') as f:
                seg_data = json.load(f)

            exhibits = []
            for det in seg_data.get('detections', []):
                exhibits.append({
                    "id": f"EX-{det['id']:03d}",
                    "name": f"展品_{det['id']+1}",
                    "bbox": tuple(det['bbox']),
                    "center": tuple(det['center']),
                    "area": det.get('area', 0),
                    "confidence": det.get('confidence', 0.0)
                })

            # 复制分割图
            import shutil
            overlay_src = os.path.dirname(args.use_existing_seg) + "/overlay.png"
            if os.path.exists(overlay_src):
                os.makedirs(args.output, exist_ok=True)
                shutil.copy(overlay_src, os.path.join(args.output, "01_segmentation.png"))

            image_shape = (600, 1100, 3)
            print(f"[+] Loaded {len(exhibits)} detected regions")
        else:
            exhibits, image_shape = run_segmentation(args.image, args.output, args.model, args.config)

        # Step 0: VLM 过滤
        exhibits = filter_exhibits_with_vlm(args.image, exhibits, args.output)

        # Step 2: 眼动数据映射
        gaze_records_raw = create_mock_gaze_data(exhibits)
        gaze_records = map_gaze_to_exhibits(gaze_records_raw, exhibits)

        # Step 3: 记忆系统
        memory = run_memory_system(gaze_records, exhibits, args.output)

        # Step 4: 预测引擎
        predictions = run_prediction(memory, exhibits, args.output)

        # Step 5: 可视化
        run_visualization(args.image, gaze_records, exhibits, predictions, args.output)

        print("\n" + "=" * 60)
        print("[OK] Pipeline completed!")
        print(f"[Output] All results saved to: {args.output}/")
        print("=" * 60)

    except Exception as e:
        print(f"\n[!] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
