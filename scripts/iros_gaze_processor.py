#!/usr/bin/env python3
"""
IROS Gaze Processor - 眼动追踪处理系统

整合 SAM2 分割 + VLM 识别 + 热力图 + 轨迹可视化
提供统一的命令行接口供 Node.js 后端调用
"""

import sys
import os
import json
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from skills.segmentation.sam2_local import SAM2Segmenter
from skills.topology.vlm_mapper import VLMTopologyMapper
from skills.visualization.heatmap import HeatmapVisualizer
from skills.visualization.gaze_trajectory import TrajectoryVisualizer
from skills.topology.graph_engine import TopologyEngine
from config import ModelConfig


def process_gaze_analysis(
    image_path: str,
    output_dir: str,
    mask_prompt: str = "auto",
    vlm_prompt: str = "auto",
    config: dict = None
):
    """
    完整的眼动追踪分析流程
    
    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        mask_prompt: SAM2 分割提示
        vlm_prompt: VLM 识别提示
        config: 配置字典
    
    Returns:
        结果字典
    """
    print(f"[1/5] 加载配置...")
    model_config = ModelConfig(**config) if config else ModelConfig()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    trajectory_path = os.path.join(output_dir, "trajectory.png")
    
    print(f"[2/5] 执行 SAM2 语义分割...")
    segmenter = SAM2Segmenter(model_config)
    segmentation_result = segmenter.segment_image(image_path, mask_prompt)
    
    if segmentation_result.get("error"):
        print(f"❌ 分割失败：{segmentation_result['error']}")
        return {"error": segmentation_result["error"]}
    
    print(f"[3/5] 调用 VLM 进行场景识别...")
    vlm_mapper = VLMTopologyMapper(model_config)
    topology_result = vlm_mapper.recognize_topology(
        image_path,
        segmentation_result,
        prompt=vlm_prompt
    )
    
    if topology_result.get("error"):
        print(f"⚠️  VLM 识别警告：{topology_result['error']}")
    
    print(f"[4/5] 生成热力图...")
    try:
        # 创建拓扑引擎
        topology_engine = TopologyEngine()
        if topology_result.get("exhibits"):
            topology_engine.build_from_exhibits(topology_result["exhibits"])
        
        # 生成访问热力图
        heatmap_viz = HeatmapVisualizer(topology_engine)
        
        # 模拟访问数据（实际应从眼动数据获取）
        visit_counts = {
            exhibit["id"]: exhibit.get("visit_count", 1)
            for exhibit in topology_result.get("exhibits", [])
        }
        
        fig = heatmap_viz.plot_visit_heatmap(
            visit_counts,
            output_path=heatmap_path,
            show=False
        )
        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f"✅ 热力图已保存：{heatmap_path}")
    except Exception as e:
        print(f"⚠️  热力图生成失败：{e}")
        heatmap_path = None
    
    print(f"[5/5] 生成眼动轨迹...")
    try:
        trajectory_viz = TrajectoryVisualizer(topology_engine)
        
        # 模拟眼动序列（实际应从预测模型获取）
        from skills.memory.manager import GazeRecord
        gaze_sequence = []
        for i, exhibit in enumerate(topology_result.get("exhibits", [])[:5]):
            gaze_sequence.append(GazeRecord(
                exhibit_id=exhibit["id"],
                timestamp=i * 1000,
                duration=2000,
                confidence=0.9
            ))
        
        fig = trajectory_viz.plot_gaze_trajectory(
            gaze_sequence,
            output_path=trajectory_path,
            show=False
        )
        fig.savefig(trajectory_path, dpi=150, bbox_inches='tight')
        print(f"✅ 轨迹图已保存：{trajectory_path}")
    except Exception as e:
        print(f"⚠️  轨迹图生成失败：{e}")
        trajectory_path = None
    
    # 返回结果
    result = {
        "success": True,
        "segmentation": {
            "num_masks": len(segmentation_result.get("masks", [])),
            "scores": segmentation_result.get("scores", [])
        },
        "topology": topology_result,
        "outputs": {
            "heatmap": heatmap_path,
            "trajectory": trajectory_path
        }
    }
    
    print("\n✅ 分析完成!")
    return result


def main():
    parser = argparse.ArgumentParser(description='IROS Gaze 眼动追踪处理系统')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--mask-prompt', type=str, default='auto', help='SAM2 分割提示')
    parser.add_argument('--vlm-prompt', type=str, default='auto', help='VLM 识别提示')
    parser.add_argument('--config', type=str, help='配置文件路径 (JSON)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 执行处理
    result = process_gaze_analysis(
        image_path=args.image,
        output_dir=args.output,
        mask_prompt=args.mask_prompt,
        vlm_prompt=args.vlm_prompt,
        config=config
    )
    
    # 输出 JSON 结果
    print("\n" + "="*60)
    print("JSON RESULT:")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
