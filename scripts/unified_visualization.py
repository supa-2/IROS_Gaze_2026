#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一眼动可视化脚本

整合 VLM 识别、热力图、轨迹图的完整工作流
用于 IROS 论文展示

使用方式:
    python scripts/unified_visualization.py --image data/R.jpg
    python scripts/unified_visualization.py --image data/R.jpg --use-sam2
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from skills.visualization.pixel_heatmap import GazeHeatmapVisualizer, GazeRegion
from skills.visualization.gaze_trajectory import GazeTrajectoryVisualizer
from config import Config


# ============================================
# VLM 展品识别模块
# ============================================

class VLMRecognizer:
    """VLM 展品识别器"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """初始化 VLM 识别器"""
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or os.getenv("VLM_MODEL", "qwen-vl-max-latest")

    def recognize(self, image_path: str) -> Optional[List[Dict]]:
        """
        使用 VLM 识别展品

        Returns:
            展品列表: [{"id": str, "name": str, "type": str, "bbox": [x1,y1,x2,y2], "description": str}, ...]
        """
        import base64
        import re

        print("\n" + "="*70)
        print("[Step 1/4] VLM 展品识别")
        print("="*70)

        if not self.api_key or self.api_key == "your_api_key_here":
            print("[!] 错误: 未找到有效的 API Key")
            print("    请在 .env 文件中设置 QWEN_API_KEY")
            return None

        print(f"[*] API: {self.base_url}")
        print(f"[*] Model: {self.model}")

        # 编码图片
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        # 获取图片尺寸
        img = Image.open(image_path)
        width, height = img.size

        prompt = f"""请分析这张展厅图片，识别出所有值得观看的展品。

对于每个展品，请提供：
1. 展品名称（如：画作1、雕塑A）
2. 展品类型（如：画作、雕塑、装置艺术、说明牌）
3. 在图片中的位置（边界框坐标 [x1, y1, x2, y2]，其中 (0,0) 是左上角）
4. 简短描述

图片尺寸: {width} x {height}

请以 JSON 格式返回：
[
  {{
    "name": "展品名称",
    "type": "展品类型",
    "bbox": [x1, y1, x2, y2],
    "description": "简短描述"
  }}
]

要求：
- 识别所有主要的展品（画作、雕塑、装置等）
- 边界框要紧凑地包围展品主体
- 坐标范围：x: 0-{width}, y: 0-{height}
- 对于说明牌，也请识别
- 返回 5-15 个展品"""

        print("[*] 调用 VLM API...")

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = client.chat.completions.create(
                model=self.model,
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
                max_tokens=2500
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
                            # 验证坐标范围
                            if 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height:
                                area = (x2 - x1) * (y2 - y1)
                                # 过滤过小或过大的区域
                                if 500 < area < width * height * 0.5:
                                    valid_exhibits.append({
                                        "id": f"EX-{i:03d}",
                                        "name": ex.get('name', f'展品{i+1}'),
                                        "type": ex.get('type', 'Unknown'),
                                        "bbox": [x1, y1, x2, y2],
                                        "description": ex.get('description', '')
                                    })
                        except (ValueError, TypeError):
                            continue

                print(f"[+] 识别到 {len(valid_exhibits)} 个有效展品")
                for ex in valid_exhibits:
                    print(f"    - {ex['name']} ({ex['type']}) at {ex['bbox']}")
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
# 眼动数据生成模块
# ============================================

class GazeDataGenerator:
    """眼动数据生成器"""

    def __init__(self, attention_config: Dict = None):
        """
        初始化眼动数据生成器

        Args:
            attention_config: 注意力等级配置 {level: duration_seconds}
        """
        self.attention_config = attention_config or {
            'A': 120,  # 深度关注
            'B': 60,   # 中等关注
            'C': 30,   # 一般关注
            'D': 15,   # 快速浏览
            'E': 5     # 一瞥
        }

    def generate(
        self,
        exhibits: List[Dict],
        num_records: int = 30,
        random_seed: int = None
    ) -> List[Dict]:
        """
        生成模拟眼动数据

        Args:
            exhibits: 展品列表
            num_records: 生成记录数
            random_seed: 随机种子

        Returns:
            眼动记录列表
        """
        if random_seed is not None:
            import random
            random.seed(random_seed)

        print("\n" + "="*70)
        print("[Step 2/4] 生成眼动数据")
        print("="*70)

        import random

        gaze_records = []
        attention_levels = list(self.attention_config.keys())

        # 模拟真实观看模式
        num_exhibits = len(exhibits)

        # 生成访问序列（包含重复观看）
        visit_pattern = []
        for i in range(num_records):
            if i < num_exhibits:
                # 首次遍历所有展品
                visit_pattern.append(i)
            else:
                # 随机回看之前的展品
                visit_pattern.append(random.randint(0, min(i, num_exhibits - 1)))

        for i, exhibit_idx in enumerate(visit_pattern):
            exhibit = exhibits[exhibit_idx]
            bbox = exhibit['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            # 添加随机偏移（模拟凝视点不总是在中心）
            offset_range = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 4
            x = center_x + random.randint(-offset_range, offset_range)
            y = center_y + random.randint(-offset_range, offset_range)

            # 随机选择注意力等级
            level = random.choice(attention_levels)
            base_duration = self.attention_config[level]
            duration = base_duration * random.uniform(0.5, 1.5)

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
        print(f"[*] 访问的展品数: {len(set(r['exhibit_id'] for r in gaze_records))}")
        print(f"[*] 总观看时长: {sum(r['duration'] for r in gaze_records):.1f}秒")

        return gaze_records


# ============================================
# 统一可视化器
# ============================================

class UnifiedVisualizer:
    """统一可视化器 - 整合热力图和轨迹图"""

    def __init__(self, image_path: str, output_dir: str):
        """
        初始化统一可视化器

        Args:
            image_path: 展厅图片路径
            output_dir: 输出目录
        """
        self.image_path = image_path
        self.output_dir = output_dir
        self.original_image = Image.open(image_path).convert('RGB')
        self.width, self.height = self.original_image.size

        os.makedirs(output_dir, exist_ok=True)

        # 初始化热力图可视化器
        self.heatmap_viz = GazeHeatmapVisualizer(image_path, sigma=50)

        # 初始化轨迹可视化器
        self.trajectory_viz = GazeTrajectoryVisualizer(image_path)

        # 展品区域
        self.exhibits: List[Dict] = []

        # 眼动记录
        self.gaze_records: List[Dict] = []

    def add_exhibits(self, exhibits: List[Dict]):
        """添加展品区域"""
        self.exhibits = exhibits

        # 添加到热力图可视化器
        for ex in exhibits:
            region = GazeRegion(
                id=ex['id'],
                label=ex['name'],
                type=ex.get('type', 'Exhibit'),
                bbox=tuple(ex['bbox'])
            )
            self.heatmap_viz.regions[region.id] = region

        # 添加到轨迹可视化器
        for ex in exhibits:
            self.trajectory_viz.add_region(
                region_id=ex['id'],
                label=ex['name'],
                bbox=ex['bbox']
            )

        print(f"[*] 添加了 {len(exhibits)} 个展品区域")

    def add_gaze_records(self, gaze_records: List[Dict]):
        """添加眼动记录"""
        self.gaze_records = gaze_records

        # 添加到热力图可视化器
        for record in gaze_records:
            self.heatmap_viz.add_gaze_record(
                region_id=record['exhibit_id'],
                duration=record['duration'],
                x=record['x'],
                y=record['y']
            )

        # 添加到轨迹可视化器
        for record in gaze_records:
            self.trajectory_viz.add_trajectory_point(
                x=record['x'],
                y=record['y'],
                duration=record['duration'],
                region_id=record['exhibit_id'],
                sequence_number=record['sequence']
            )

        print(f"[*] 添加了 {len(gaze_records)} 条眼动记录")

    def draw_vlm_results(self, output_path: str = None) -> str:
        """绘制 VLM 识别结果"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "01_vlm_recognition.png")

        img = self.original_image.copy()
        draw = ImageDraw.Draw(img)

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            (0, 128, 128), (128, 128, 0)
        ]

        for i, ex in enumerate(self.exhibits):
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
        print(f"[*] VLM识别结果保存到: {output_path}")
        return output_path

    def generate_heatmap(self) -> Tuple[str, Dict]:
        """生成热力图"""
        print("\n" + "="*70)
        print("[Step 3/4] 生成热力图")
        print("="*70)

        # 计算热力图
        self.heatmap_viz.calculate_heatmap()

        # 获取统计
        stats = self.heatmap_viz.get_region_statistics()
        print(f"[*] 眼动统计: {len(stats)} 个区域有凝视数据")
        for rid, rstat in stats.items():
            print(f"    {rstat['label']}: {rstat['fixation_count']} 次凝视, {rstat['total_duration']:.1f}s")

        # 保存统计
        stats_path = os.path.join(self.output_dir, "heatmap_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # 生成并排对比图
        comparison_path = os.path.join(self.output_dir, "02_heatmap_comparison.png")
        self.heatmap_viz.visualize_side_by_side(comparison_path)

        # 生成叠加图
        overlay_path = os.path.join(self.output_dir, "03_heatmap_overlay.png")
        self.heatmap_viz.visualize_overlay(overlay_path, alpha=0.6)

        return overlay_path, stats

    def generate_trajectory(self, heatmap_data: np.ndarray = None) -> Tuple[str, Dict]:
        """生成轨迹图"""
        print("\n" + "="*70)
        print("[Step 4/4] 生成轨迹图")
        print("="*70)

        # 生成轨迹图
        trajectory_path = os.path.join(self.output_dir, "04_gaze_trajectory.png")
        self.trajectory_viz.visualize(trajectory_path)

        # 生成轨迹+热力图叠加
        if heatmap_data is not None:
            combined_path = os.path.join(self.output_dir, "05_trajectory_heatmap_combined.png")
            self.trajectory_viz.visualize_with_heatmap(
                heatmap_data,
                combined_path,
                heatmap_alpha=0.5
            )

        # 统计
        traj_stats = self.trajectory_viz.get_statistics()
        print(f"[*] 轨迹统计: {traj_stats['total_fixations']} 次凝视, "
              f"{traj_stats['unique_regions']} 个区域, "
              f"{traj_stats['region_transitions']} 次转换")

        # 保存统计
        traj_stats_path = os.path.join(self.output_dir, "trajectory_statistics.json")
        with open(traj_stats_path, 'w', encoding='utf-8') as f:
            json.dump(traj_stats, f, indent=2, ensure_ascii=False)

        return trajectory_path, traj_stats

    def generate_final_report(self) -> Dict:
        """生成最终报告"""
        # 计算总统计
        total_duration = sum(r['duration'] for r in self.gaze_records)
        avg_duration = total_duration / len(self.gaze_records) if self.gaze_records else 0

        level_counts = {}
        for r in self.gaze_records:
            level = r['attention_level']
            level_counts[level] = level_counts.get(level, 0) + 1

        report = {
            "timestamp": datetime.now().isoformat(),
            "image_path": self.image_path,
            "output_dir": self.output_dir,
            "summary": {
                "total_exhibits": len(self.exhibits),
                "total_gaze_records": len(self.gaze_records),
                "unique_exhibits_visited": len(set(r['exhibit_id'] for r in self.gaze_records)),
                "total_duration": round(total_duration, 1),
                "average_duration": round(avg_duration, 1),
                "attention_level_distribution": level_counts
            },
            "exhibits": self.exhibits,
            "gaze_records": self.gaze_records
        }

        # 保存报告
        report_path = os.path.join(self.output_dir, "FINAL_REPORT.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印摘要
        print("\n" + "="*70)
        print("【最终报告】")
        print("="*70)
        print(f"检测展品数: {report['summary']['total_exhibits']}")
        print(f"眼动记录数: {report['summary']['total_gaze_records']}")
        print(f"访问展品数: {report['summary']['unique_exhibits_visited']}")
        print(f"总观看时长: {report['summary']['total_duration']:.1f}秒")
        print(f"平均停留: {report['summary']['average_duration']:.1f}秒")
        print(f"注意力分布: {level_counts}")
        print(f"\n输出目录: {self.output_dir}/")
        print("="*70)

        return report


# ============================================
# 主流程
# ============================================

def run_unified_pipeline(
    image_path: str,
    output_dir: str = None,
    use_vlm: bool = True,
    vlm_result_path: str = None,
    num_records: int = 30,
    random_seed: int = 42
) -> Dict:
    """
    运行统一可视化流程

    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        use_vlm: 是否使用VLM识别
        vlm_result_path: 已有VLM结果路径
        num_records: 生成眼动记录数
        random_seed: 随机种子
    """
    print("\n" + "="*70)
    print("Eye-LLM 统一眼动可视化系统")
    print("="*70)
    print(f"输入: {image_path}")

    # 确定输出目录
    if output_dir is None:
        output_dir = f"data/outputs/unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 初始化可视化器
    visualizer = UnifiedVisualizer(image_path, output_dir)

    # 步骤1: VLM识别
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

    elif use_vlm:
        # 运行VLM识别
        recognizer = VLMRecognizer()
        exhibits = recognizer.recognize(image_path)

        if exhibits:
            # 保存VLM结果
            vlm_result = {
                "image_path": image_path,
                "num_exhibits": len(exhibits),
                "exhibits": exhibits
            }
            vlm_path = os.path.join(output_dir, "vlm_result.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(vlm_path, 'w', encoding='utf-8') as f:
                json.dump(vlm_result, f, indent=2, ensure_ascii=False)

    if not exhibits:
        print("[!] 错误: 没有可用的展品数据")
        return None

    # 添加展品到可视化器
    visualizer.add_exhibits(exhibits)

    # 绘制VLM识别结果
    visualizer.draw_vlm_results()

    # 步骤2: 生成眼动数据
    generator = GazeDataGenerator()
    gaze_records = generator.generate(exhibits, num_records=num_records, random_seed=random_seed)

    # 保存眼动数据
    gaze_path = os.path.join(output_dir, "gaze_records.json")
    with open(gaze_path, 'w', encoding='utf-8') as f:
        json.dump(gaze_records, f, indent=2, ensure_ascii=False)

    # 添加眼动记录到可视化器
    visualizer.add_gaze_records(gaze_records)

    # 步骤3: 生成热力图
    _, heatmap_stats = visualizer.generate_heatmap()

    # 步骤4: 生成轨迹图
    _, traj_stats = visualizer.generate_trajectory(visualizer.heatmap_viz.heatmap_data)

    # 步骤5: 生成最终报告
    report = visualizer.generate_final_report()

    print("\n" + "="*70)
    print("[完成] 所有可视化已生成!")
    print(f"输出目录: {output_dir}/")
    print("="*70)

    return {
        'output_dir': output_dir,
        'exhibits': exhibits,
        'gaze_records': gaze_records,
        'report': report
    }


def main():
    parser = argparse.ArgumentParser(description="统一眼动可视化系统")
    parser.add_argument("--image", type=str, default="data/R.jpg",
                       help="输入图片路径")
    parser.add_argument("--output", type=str, default=None,
                       help="输出目录")
    parser.add_argument("--vlm-result", type=str, default=None,
                       help="VLM识别结果JSON路径")
    parser.add_argument("--no-vlm", action="store_true",
                       help="不运行VLM识别，使用已有结果")
    parser.add_argument("--num-records", type=int, default=30,
                       help="生成眼动记录数")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] 错误: 图片不存在: {args.image}")
        return

    # 如果没有指定VLM结果路径，尝试自动查找
    vlm_result_path = args.vlm_result
    if vlm_result_path is None and args.no_vlm:
        possible_paths = [
            "data/outputs/vlm/vlm_result.json",
            "data/outputs/unified_*/vlm_result.json"
        ]
        for path in possible_paths:
            if '*' in path:
                import glob
                matches = glob.glob(path)
                if matches:
                    vlm_result_path = matches[-1]  # 使用最新的
                    break
            else:
                if os.path.exists(path):
                    vlm_result_path = path
                    break

        if vlm_result_path:
            print(f"[*] 自动找到VLM结果: {vlm_result_path}")

    run_unified_pipeline(
        image_path=args.image,
        output_dir=args.output,
        use_vlm=not args.no_vlm and vlm_result_path is None,
        vlm_result_path=vlm_result_path,
        num_records=args.num_records,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
