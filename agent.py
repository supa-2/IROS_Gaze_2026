"""
Eye-LLM Agent - Enhanced with New Architecture

This module integrates the new prediction engine and memory management
systems with the existing agent functionality.
"""

import sys
import os
from typing import List, Dict, Optional
from datetime import datetime

# Import original components
sys.path.append(os.path.join(os.path.dirname(__file__), 'skills', 'topology'))
from graph_engine import TopologyEngine

# Import new architecture components
from config import Config
from skills.prediction.engine import PredictionEngine
from skills.memory.manager import GazeRecord
from skills.visualization.heatmap import HeatmapVisualizer
from skills.visualization.trajectory import TrajectoryVisualizer
from skills.visualization.network import NetworkVisualizer


class EyeLLMAgent:
    """
    Eye-LLM Agent - Enhanced with new architecture

    This agent now includes:
    - PredictionEngine for advanced predictions
    - Memory management (short-term and long-term)
    - Visualization capabilities
    - Backward compatibility with original methods
    """

    def __init__(self, map_name='TH', model_name='gpt-4o', use_new_architecture=True):
        """
        初始化Agent

        Args:
            map_name: 地图名称 (默认'TH')
            model_name: LLM模型名称
            use_new_architecture: 是否使用新架构 (默认True)
        """
        self.map_name = map_name
        self.use_new_architecture = use_new_architecture

        # 初始化拓扑引擎（始终需要）
        self.topology = TopologyEngine(map_name)

        # 使用新架构
        if use_new_architecture:
            try:
                self.config = Config()
                # Override model name if provided
                if model_name:
                    self.config.model.llm_model = model_name

                self.prediction_engine = PredictionEngine(self.config, map_name)

                # 初始化可视化器
                self.heatmap_viz = HeatmapVisualizer(self.topology)
                self.trajectory_viz = TrajectoryVisualizer(self.topology)
                self.network_viz = NetworkVisualizer(self.topology)

                print(f"✅ Eye-LLM Agent initialized with new architecture")
                print(f"   Map: {map_name}")
                print(f"   Model: {self.config.model.llm_model}")

            except Exception as e:
                print(f"⚠️  Failed to initialize new architecture: {e}")
                print("   Falling back to original implementation...")
                self.use_new_architecture = False

        # 使用原始实现（作为fallback）
        if not self.use_new_architecture:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model_name)
            print(f"✅ Eye-LLM Agent initialized with original implementation")
            print(f"   Map: {map_name}")
            print(f"   Model: {model_name}")

    # ==========================================
    # 新架构方法 (New Architecture Methods)
    # ==========================================

    def add_observation(
        self,
        exhibit_id: str,
        exhibit_name: str = None,
        attention_level: str = 'C'
    ):
        """
        添加观测记录（新架构）

        Args:
            exhibit_id: 展品ID
            exhibit_name: 展品名称（如果为None则从拓扑获取）
            attention_level: 注意力等级 (A/B/C/D/E)
        """
        if not self.use_new_architecture:
            raise NotImplementedError("add_observation requires new architecture")

        # 从拓扑获取展品名称（如果未提供）
        if exhibit_name is None:
            node_data = self.topology.query_node(exhibit_id)
            if "error" not in node_data:
                exhibit_name = node_data['info']['name']
            else:
                exhibit_name = f"Exhibit {exhibit_id}"

        self.prediction_engine.add_observation(
            exhibit_id=exhibit_id,
            exhibit_name=exhibit_name,
            attention_level=attention_level
        )

        print(f"✅ Added observation: {exhibit_name} ({exhibit_id}) - Level {attention_level}")

    def predict_next(self, verbose: bool = True) -> Dict:
        """
        预测下一个目标（新架构）

        Args:
            verbose: 是否打印详细信息

        Returns:
            预测结果字典
        """
        if not self.use_new_architecture:
            raise NotImplementedError("predict_next requires new architecture")

        return self.prediction_engine.predict_next(verbose=verbose)

    def predict_sequence(self, n_steps: int = 5, verbose: bool = True) -> List[Dict]:
        """
        预测未来序列（新架构）

        Args:
            n_steps: 预测步数
            verbose: 是否打印详细信息

        Returns:
            预测序列列表
        """
        if not self.use_new_architecture:
            raise NotImplementedError("predict_sequence requires new architecture")

        return self.prediction_engine.predict_sequence(n_steps=n_steps, verbose=verbose)

    def get_memory_stats(self) -> Dict:
        """获取记忆统计信息"""
        if not self.use_new_architecture:
            raise NotImplementedError("get_memory_stats requires new architecture")

        return self.prediction_engine.get_memory_statistics()

    def visualize_trajectory(self, output_dir: str = "data/outputs/trajectories"):
        """
        可视化历史轨迹

        Args:
            output_dir: 输出目录
        """
        if not self.use_new_architecture:
            raise NotImplementedError("visualize_trajectory requires new architecture")

        history = self.prediction_engine.get_recent_history()

        if not history:
            print("⚠️  No history to visualize")
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/historical_{timestamp}.png"

        self.trajectory_viz.plot_gaze_trajectory(history, output_path=output_path)
        print(f"✅ Trajectory visualization saved to: {output_path}")

    def visualize_heatmap(self, output_dir: str = "data/outputs/heatmaps"):
        """
        可视化访问热力图

        Args:
            output_dir: 输出目录
        """
        if not self.use_new_architecture:
            raise NotImplementedError("visualize_heatmap requires new architecture")

        stats = self.prediction_engine.get_memory_statistics()

        # 构建访问次数字典
        visit_counts = {}
        duration_data = {}

        for record in self.prediction_engine.memory.get_all_history():
            exhibit_id = record.exhibit_id
            visit_counts[exhibit_id] = visit_counts.get(exhibit_id, 0) + 1

            duration = record.actual_duration if record.actual_duration else record.estimated_duration
            duration_data[exhibit_id] = duration_data.get(exhibit_id, 0) + duration

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 访问频率热力图
        if visit_counts:
            visit_path = f"{output_dir}/visit_heatmap_{timestamp}.png"
            self.heatmap_viz.plot_visit_heatmap(visit_counts, output_path=visit_path)
            print(f"✅ Visit frequency heatmap saved to: {visit_path}")

        # 停留时间热力图
        if duration_data:
            duration_path = f"{output_dir}/duration_heatmap_{timestamp}.png"
            self.heatmap_viz.plot_duration_heatmap(duration_data, output_path=duration_path)
            print(f"✅ Duration heatmap saved to: {duration_path}")

    # ==========================================
    # 原始方法 (保持向后兼容)
    # ==========================================

    def _format_path(self, path_list):
        """将路径列表格式化为易读字符串"""
        if not path_list:
            return "无"
        return " -> ".join([f"{n['name']}({n['id']})" for n in path_list])

    def predict_next_gaze(self, gaze_history: List[str]):
        """
        核心功能：基于历史预测未来（原始实现）

        Args:
            gaze_history: 展品ID列表

        Returns:
            预测结果字符串
        """
        if self.use_new_architecture:
            # 使用新架构实现
            # 清空记忆并添加历史记录
            self.prediction_engine.clear_memory()

            for exhibit_id in gaze_history:
                node_data = self.topology.query_node(exhibit_id)
                if "error" not in node_data:
                    name = node_data['info']['name']
                    # 默认使用中等注意力
                    self.add_observation(exhibit_id, name, 'C')

            # 预测
            result = self.predict_next(verbose=False)

            # 格式化输出
            if 'error' in result:
                return f"Error: {result['error']}"
            else:
                return {
                    'prediction_id': result['prediction_id'],
                    'prediction_name': result['prediction_name'],
                    'reasoning': result.get('reasoning', ''),
                    'confidence': result.get('confidence', 0.0),
                    'attention_level': result.get('attention_level', 'C')
                }

        # 原始实现（fallback）
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        current_id = gaze_history[-1]
        context_data = self.topology.query_node(current_id)

        if "error" in context_data:
            return f"Error: {context_data['error']}"

        ctx = context_data.get('context', {})
        prev_path = self._format_path(ctx.get('previous_path', []))
        next_path = self._format_path(ctx.get('next_path', []))

        choices = []
        for choice in ctx.get('direct_choices', []):
            node_info = self.topology.graph.nodes.get(choice['id'], {})
            choices.append(f"- ID: {choice['id']} | 名称: {node_info.get('name')} | 关系: {choice['relation']}")
        choices_str = "\n".join(choices)

        system_prompt = """你是一个专业的博物馆空间行为分析专家 (Space Syntax Expert)。
你的任务是根据用户的"视觉历史"和当前的"空间拓扑约束"，预测用户的下一个视觉关注点 (Gaze Fixation)。

请遵循以下分析逻辑：
1. **语义连贯性 (Semantic Continuity)**: 用户是否正在阅读一系列相关的内容（如说明牌 -> 展品）？
2. **空间流线 (Spatial Flow)**: 用户是否顺应了展厅设计的 'Next' 动线？
3. **视觉显著性 (Visual Saliency)**: 如果有 'Visual' 类型的关联，且距离很近，用户可能会被吸引。

请只输出 JSON 格式结果，包含 'prediction_id' 和 'reasoning' 两个字段。
"""

        user_prompt = f"""
【用户轨迹】
过去路径: {gaze_history}
当前位置: {context_data['info']['name']} ({current_id})
当前位置特征: {context_data['info']['features']}

【拓扑上下文 (Environment Context)】
- 动线前序 (来源于): {prev_path}
- 动线后续 (去往): {next_path}

【当前物理可达选项 (Affordances)】
{choices_str}

请预测下一个 ID。
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        print(f"Thinking... (Current: {current_id})")
        result = chain.invoke({})
        return result


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    import json

    print("\n" + "="*70)
    print("  Eye-LLM Agent - Enhanced with New Architecture")
    print("="*70 + "\n")

    # 实例化 Agent (使用清华地图)
    # 注意：请确保在.env文件中配置了正确的API密钥
    agent = EyeLLMAgent(map_name='TH', use_new_architecture=True)

    if agent.use_new_architecture:
        print("\n📊 Using NEW architecture with PredictionEngine\n")
        print("="*70)

        # 模拟用户轨迹
        print("\n📍 Adding observations...")
        agent.add_observation("TH-E01", attention_level='A')
        agent.add_observation("TH-I-B01", attention_level='B')
        agent.add_observation("TH-B02", attention_level='A')

        print("\n📈 Memory Statistics:")
        stats = agent.get_memory_stats()
        print(f"   Total observations: {stats['long_term_count']}")
        print(f"   Unique exhibits: {stats['unique_exhibits']}")
        print(f"   Most visited: {stats['most_visited'][:3]}")

        print("\n🔮 Predicting next exhibit...")
        prediction = agent.predict_next(verbose=True)

        if 'error' not in prediction:
            print("\n🎯 Prediction Result:")
            print(f"   Next: {prediction['prediction_name']} ({prediction['prediction_id']})")
            print(f"   Attention: {prediction['attention_level']}")
            print(f"   Duration: {prediction['estimated_duration']}s")
            print(f"   Confidence: {prediction.get('confidence', 0.0):.2f}")

            print("\n🔮 Predicting full sequence (5 steps)...")
            sequence = agent.predict_sequence(n_steps=5, verbose=True)

            # 导出预测结果
            print("\n💾 Exporting predictions...")
            agent.prediction_engine.export_predictions(
                sequence,
                output_path="data/outputs/predictions/test_prediction.json"
            )

            print("\n🎨 Generating visualizations...")
            agent.visualize_trajectory()
            agent.visualize_heatmap()

    else:
        print("\n📊 Using ORIGINAL architecture\n")
        print("="*70)

        # 模拟一段用户历史：入口 -> 入口说明牌 -> 原始森林画
        fake_history = ['TH-E01', 'TH-I-B01', 'TH-B02']

        print("\n📍 Simulating User Path:", fake_history)
        print("-" * 30)

        prediction = agent.predict_next_gaze(fake_history)

        print("\n🎯 Prediction Result:")
        print(prediction)

    print("\n" + "="*70)
    print("  Agent Test Complete")
    print("="*70 + "\n")
