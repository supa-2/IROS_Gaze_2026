#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation Image Pipeline Script

功能:
1. 监控 data/simulation/ 文件夹获取最新上传的图片
2. 使用 VLM (Qwen) 识别图片中的展品
3. 构建展品拓扑关系和 memory
4. 传给 gaze 预测模型预测轨迹和停留时间

用法:
    python scripts/simulation_pipeline.py --watch              # 持续监控模式
    python scripts/simulation_pipeline.py --once               # 单次运行模式
    python scripts/simulation_pipeline.py --image path/to.jpg  # 指定图片
"""

import os
import sys
import json
import time
import base64
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from openai import OpenAI
from config import Config

# 导入动态拓扑引擎
sys.path.insert(0, str(project_root / "skills" / "topology"))
from dynamic_topology import DynamicTopologyEngine

# 导入预测引擎组件
from skills.memory.manager import MemoryManager
from skills.prediction.context_builder import ContextBuilder
from skills.prediction.llm_reasoner import LLMReasoner
from skills.prediction.sequence_predictor import SequencePredictor


# ===========================
# 配置
# ===========================
SIMULATION_DIR = project_root / "data" / "simulation"
OUTPUT_DIR = project_root / "data" / "outputs" / "simulation"
CHECK_INTERVAL = 5  # 监控检查间隔（秒）


class SimulationImageWatcher:
    """监控文件夹中的新图片"""

    def __init__(self, watch_dir: Path):
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.last_processed = None
        self.last_mtime = 0

    def get_latest_image(self) -> Optional[str]:
        """获取最新的图片文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        image_files = [
            f for f in self.watch_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            return None

        # 按修改时间排序
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest = image_files[0]

        # 检查是否有新文件
        if latest.stat().st_mtime > self.last_mtime:
            self.last_mtime = latest.stat().st_mtime
            self.last_processed = str(latest)
            return str(latest)

        return None


class VLMExhibitRecognizer:
    """使用 VLM 识别图片中的展品"""

    def __init__(self, config: Config):
        self.config = config

        # 优先使用 Qwen API
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.getenv("VLM_MODEL", "qwen-vl-max-latest")

        if not api_key or api_key == "your_api_key_here":
            raise ValueError("未找到有效的 API Key，请在 .env 文件中设置 QWEN_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        print(f"[VLM] API: {base_url}")
        print(f"[VLM] Model: {model}")

    def encode_image(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def recognize_exhibits(self, image_path: str) -> List[Dict]:
        """
        识别图片中的展品

        返回格式:
        [
            {
                "id": "exhibit_1",
                "name": "展品名称",
                "type": "类型",
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
                "description": "描述",
                "visual_features": "视觉特征"
            }
        ]
        """
        print(f"\n[VLM] 正在识别: {image_path}")

        image_base64 = self.encode_image(image_path)

        prompt = """请分析这张博物馆展厅图片，识别出所有值得观看的展品。

对于每个展品，请提供:
1. id: 唯一标识符 (如 "E1", "E2", "E3"... 从左到右，从上到下编号)
2. name: 展品名称 (如 "古代青铜鼎"、"山水画作"、"抽象雕塑")
3. type: 展品类型 (如 "painting"=绘画, "sculpture"=雕塑, "artifact"=文物, "label"=说明牌)
4. bbox: 边界框坐标 [x1, y1, x2, y2]，左上角为(0,0)，右下角为图片宽高
5. center: 中心点坐标 [cx, cy]
6. description: 简短描述 (材质、年代、风格等，20-50字)
7. visual_features: 视觉特征 (颜色、形状、大小、位置等)
8. attention_level: 预估观众关注度 (A=深度关注, B=中等关注, C=一般关注, D=快速浏览, E=一瞥)

同时分析展品之间的空间关系:
- 哪些展品相邻 (next_to)
- 哪些展品在视觉上相关 (visual)
- 推荐的观看路径顺序

请以 JSON 格式返回，格式如下:
{
    "exhibits": [
        {
            "id": "E1",
            "name": "展品名称",
            "type": "painting",
            "bbox": [x1, y1, x2, y2],
            "center": [cx, cy],
            "description": "描述",
            "visual_features": "视觉特征",
            "attention_level": "B"
        }
    ],
    "relationships": [
        {"source": "E1", "target": "E2", "relation": "next_to"},
        {"source": "E2", "target": "E3", "relation": "visual"}
    ],
    "recommended_path": ["E1", "E2", "E3"],
    "entry_point": "E1"
}

要求:
- 只识别真正的展品，忽略墙壁、地板、展柜
- 边界框要紧凑包围展品主体
- 根据展品大小、位置、显著性预估关注度
- 推荐路径应考虑自然观看顺序（从左到右、从入口开始）
"""

        try:
            response = self.client.chat.completions.create(
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
                max_tokens=3000
            )

            result_text = response.choices[0].message.content
            print(f"[VLM] 响应长度: {len(result_text)} 字符")

            # 解析 JSON（处理 markdown 代码块）
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group(1) if json_match.group(1) else json_match.group())
                print(f"[VLM] 识别到 {len(result.get('exhibits', []))} 个展品")
                return result
            else:
                print("[VLM] 无法解析 JSON 响应")
                print(f"[VLM] 原始响应: {result_text[:500]}")
                return None

        except Exception as e:
            print(f"[VLM] API 调用失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class DynamicTopologyBuilder:
    """构建动态拓扑图"""

    def __init__(self):
        self.exhibits = []
        self.relationships = []

    def from_vlm_result(self, vlm_result: Dict) -> Dict:
        """从 VLM 识别结果构建拓扑"""
        self.exhibits = vlm_result.get('exhibits', [])
        self.relationships = vlm_result.get('relationships', [])

        # 如果没有提供关系，根据位置自动推断
        if not self.relationships and self.exhibits:
            self.relationships = self._infer_spatial_relations()

        return {
            "nodes": self.exhibits,
            "edges": self.relationships,
            "entry_point": vlm_result.get('entry_point', self.exhibits[0]['id'] if self.exhibits else None),
            "recommended_path": vlm_result.get('recommended_path', [e['id'] for e in self.exhibits])
        }

    def _infer_spatial_relations(self) -> List[Dict]:
        """根据空间位置推断关系"""
        if len(self.exhibits) < 2:
            return []

        relations = []
        # 按中心点 x 坐标排序
        sorted_exhibits = sorted(self.exhibits, key=lambda e: e['center'][0])

        for i in range(len(sorted_exhibits) - 1):
            current = sorted_exhibits[i]
            next_ex = sorted_exhibits[i + 1]

            # 计算距离
            dx = next_ex['center'][0] - current['center'][0]
            dy = abs(next_ex['center'][1] - current['center'][1])
            distance = (dx ** 2 + dy ** 2) ** 0.5

            # 如果距离较近，认为是相邻
            if distance < 300:  # 像素距离阈值
                relations.append({
                    "source": current['id'],
                    "target": next_ex['id'],
                    "relation": "next_to"
                })

        return relations

    def save_topology(self, image_path: str, output_dir: Path):
        """保存拓扑数据"""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topology_file = output_dir / f"topology_{timestamp}.json"

        topology_data = {
            "timestamp": datetime.now().isoformat(),
            "source_image": image_path,
            "nodes": self.exhibits,
            "edges": self.relationships
        }

        with open(topology_file, 'w', encoding='utf-8') as f:
            json.dump(topology_data, f, ensure_ascii=False, indent=2)

        print(f"[Topology] 拓扑数据已保存: {topology_file}")
        return str(topology_file)


class GazePredictionPipeline:
    """Gaze 预测管道"""

    def __init__(self):
        self.config = Config()
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.vlm = VLMExhibitRecognizer(self.config)
        self.topology_builder = DynamicTopologyBuilder()

        # 动态拓扑引擎
        self.topology_engine = None

        # 预测引擎组件（延迟初始化）
        self.memory = None
        self.context_builder = None
        self.reasoner = None
        self.sequence_predictor = None

    def _init_prediction_components(self, vlm_result: Dict):
        """初始化预测组件（在 VLM 识别后）"""
        # 创建动态拓扑引擎
        self.topology_engine = DynamicTopologyEngine(map_name="SIMULATION")
        self.topology_engine.load_from_vlm_result(vlm_result)

        # 初始化记忆系统
        self.memory = MemoryManager(
            short_term_size=self.config.memory.short_term_size,
            long_term_max_size=self.config.memory.long_term_max_size
        )

        # 初始化上下文构建器
        self.context_builder = ContextBuilder(self.topology_engine)

        # 配置使用 Qwen API
        from config import ModelConfig
        qwen_api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        qwen_base_url = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL",
                      "https://dashscope.aliyuncs.com/compatible-mode/v1")
        llm_model = os.getenv("LLM_MODEL", "qwen3.5-plus")

        # 创建使用 Qwen 的 ModelConfig
        qwen_model_config = ModelConfig(
            llm_model=llm_model,
            llm_temperature=0.7,
            llm_max_tokens=2000,
            llm_api_key=qwen_api_key,
            llm_base_url=qwen_base_url
        )

        # 初始化 LLM 推理器（使用 Qwen 配置）
        self.reasoner = LLMReasoner(qwen_model_config)

        # 增加超时时间（Qwen API 可能响应较慢）
        try:
            self.reasoner.llm.kwargs['timeout'] = 120
        except (KeyError, AttributeError):
            pass  # 如果无法设置，使用默认值

        # 初始化序列预测器
        self.sequence_predictor = SequencePredictor(
            self.reasoner,
            self.topology_engine,
            self.config.attention
        )

        print(f"[Prediction] 预测组件已初始化 (使用 Qwen API)")

    def add_observation(self, exhibit_id: str, exhibit_name: str, attention_level: str = 'C'):
        """添加观测记录"""
        estimated_duration = self.config.attention.ATTENTION_DURATION.get(
            attention_level,
            self.config.attention.ATTENTION_DURATION['C']
        )

        self.memory.add_observation(
            exhibit_id=exhibit_id,
            exhibit_name=exhibit_name,
            attention_level=attention_level,
            estimated_duration=estimated_duration
        )

    def predict_next(self) -> Dict:
        """预测下一个目标"""
        recent = self.memory.get_recent(1)
        if not recent:
            return {"error": "No history yet"}

        current = recent[-1]
        history = self.memory.get_recent()

        context = self.context_builder.build_context(current, history)
        return self.reasoner.predict_with_explanation(
            context,
            self.config.attention,
            verbose=True
        )

    def predict_sequence(self, n_steps: int = 5) -> List[Dict]:
        """预测未来序列"""
        recent = self.memory.get_recent(1)
        if not recent:
            return []

        current = recent[-1]
        history = self.memory.get_recent()

        context = self.context_builder.build_context(current, history)
        sequence = self.sequence_predictor.predict_sequence(
            context,
            n_steps=n_steps,
            exclude_visited=False  # 对于新场景，允许重复访问
        )

        print(self.sequence_predictor.format_sequence_summary(sequence))
        return sequence

    def process_image(self, image_path: str) -> Dict:
        """处理单张图片的完整流程"""
        print(f"\n{'='*60}")
        print(f"开始处理: {image_path}")
        print(f"{'='*60}")

        # Step 1: VLM 识别展品
        vlm_result = self.vlm.recognize_exhibits(image_path)
        if not vlm_result:
            return {"error": "VLM 识别失败"}

        exhibits = vlm_result.get('exhibits', [])
        if not exhibits:
            return {"error": "未识别到任何展品"}

        # Step 2: 构建拓扑
        topology = self.topology_builder.from_vlm_result(vlm_result)
        topology_file = self.topology_builder.save_topology(image_path, self.output_dir)

        # Step 3: 初始化预测组件
        print(f"\n[Init] 初始化预测组件...")
        self._init_prediction_components(vlm_result)

        # Step 4: 添加展品到记忆（从推荐路径的第一个开始）
        print(f"\n[Memory] 添加展品到记忆系统...")
        entry_point = topology.get('entry_point') or exhibits[0]['id']
        recommended_path = topology.get('recommended_path', [e['id'] for e in exhibits])

        # 按推荐路径顺序添加前几个展品作为"已观看"
        start_count = min(2, len(recommended_path))  # 假设已经看了前2个
        for i in range(start_count):
            exhibit_id = recommended_path[i]
            exhibit = next((e for e in exhibits if e['id'] == exhibit_id), exhibits[i])
            attention_level = exhibit.get('attention_level', 'C')
            self.add_observation(
                exhibit_id=exhibit['id'],
                exhibit_name=exhibit['name'],
                attention_level=attention_level
            )
            print(f"  - 已观看: {exhibit['name']} ({exhibit['id']}) - Level {attention_level}")

        # Step 5: 预测 Gaze 轨迹
        print(f"\n[Prediction] 预测下一个目标...")
        next_prediction = self.predict_next()

        # Step 6: 预测完整序列
        print(f"\n[Prediction] 预测完整观看序列...")
        remaining_steps = len(recommended_path) - start_count
        sequence = self.predict_sequence(n_steps=max(remaining_steps, 3))

        # Step 7: 保存结果
        result = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "exhibits": exhibits,
            "topology": topology,
            "next_prediction": next_prediction,
            "predicted_sequence": sequence,
            "memory_stats": {
                "total_observations": len(self.memory.get_all_history()),
                "unique_exhibits": len(set(r.exhibit_id for r in self.memory.get_all_history()))
            }
        }

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.output_dir / f"prediction_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n[Result] 预测结果已保存: {result_file}")

        return result

    def run_once(self, image_path: str = None):
        """运行一次处理"""
        if image_path is None:
            # 获取最新图片
            watcher = SimulationImageWatcher(SIMULATION_DIR)
            image_path = watcher.get_latest_image()

        if image_path and os.path.exists(image_path):
            return self.process_image(image_path)
        else:
            print("[Watch] 未找到新图片")
            return None

    def run_watch(self):
        """持续监控模式"""
        print(f"\n[Watch] 开始监控 {SIMULATION_DIR}")
        print(f"[Watch] 按 Ctrl+C 停止\n")

        watcher = SimulationImageWatcher(SIMULATION_DIR)

        try:
            while True:
                latest_image = watcher.get_latest_image()

                if latest_image:
                    print(f"\n[Watch] 发现新图片: {latest_image}")
                    self.process_image(latest_image)
                    print(f"\n[Watch] 等待下一个图片...")
                else:
                    print(f"[Watch] 无新图片，{CHECK_INTERVAL}秒后重试...")

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n[Watch] 监控已停止")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Simulation Image Pipeline")
    parser.add_argument("--watch", action="store_true", help="持续监控模式")
    parser.add_argument("--once", action="store_true", help="单次运行模式")
    parser.add_argument("--image", type=str, help="指定图片路径")

    args = parser.parse_args()

    # 检查 API Key
    if not os.getenv("QWEN_API_KEY"):
        print("[!] 错误: 未设置 QWEN_API_KEY")
        print("    请在 .env 文件中设置 QWEN_API_KEY")
        return

    pipeline = GazePredictionPipeline()

    if args.watch:
        pipeline.run_watch()
    elif args.image:
        pipeline.run_once(args.image)
    elif args.once:
        pipeline.run_once()
    else:
        # 默认：单次运行
        print("用法:")
        print("  python scripts/simulation_pipeline.py --watch              # 持续监控")
        print("  python scripts/simulation_pipeline.py --once               # 单次运行")
        print("  python scripts/simulation_pipeline.py --image path/to.jpg  # 指定图片")
        print("\n使用默认模式（单次运行）...\n")
        pipeline.run_once()


if __name__ == "__main__":
    main()
