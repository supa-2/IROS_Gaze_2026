"""
Prediction Engine - 预测引擎总成

This module is the main orchestrator that coordinates all prediction components
including memory management, context building, LLM reasoning, and sequence prediction.
"""

from typing import List, Dict, Optional
from datetime import datetime
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'topology'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from graph_engine import TopologyEngine
from ..memory.manager import MemoryManager
from ..memory.records import GazeRecord
from .context_builder import ContextBuilder
from .llm_reasoner import LLMReasoner
from .sequence_predictor import SequencePredictor
from config import Config


class PredictionEngine:
    """
    预测引擎 - 整合所有预测组件

    This is the main interface for the prediction system. It coordinates
    memory, topology, context building, and LLM reasoning to provide
    comprehensive spatial intent predictions.
    """

    def __init__(self, config: Config, map_name: str = 'TH'):
        """
        初始化预测引擎

        Args:
            config: Config对象，包含所有配置
            map_name: 地图名称 (默认'TH')
        """
        self.config = config

        # 初始化各组件
        self.topology = TopologyEngine(map_name)
        self.memory = MemoryManager(
            short_term_size=config.memory.short_term_size,
            long_term_max_size=config.memory.long_term_max_size
        )
        self.context_builder = ContextBuilder(self.topology)
        self.reasoner = LLMReasoner(config.model)
        self.sequence_predictor = SequencePredictor(
            self.reasoner,
            self.topology,
            config.attention
        )

        # 地图名称
        self.map_name = map_name

    def add_observation(
        self,
        exhibit_id: str,
        exhibit_name: str,
        attention_level: str = 'C',
        estimated_duration: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> GazeRecord:
        """
        添加新的观测记录

        Args:
            exhibit_id: 展品ID
            exhibit_name: 展品名称
            attention_level: 注意力等级 (A/B/C/D/E, 默认'C')
            estimated_duration: 预计停留时间(秒)，如果为None则从配置获取
            timestamp: 时间戳，默认为当前时间

        Returns:
            创建的GazeRecord对象
        """
        # 获取停留时间
        if estimated_duration is None:
            estimated_duration = self.config.attention.ATTENTION_DURATION.get(
                attention_level,
                self.config.attention.ATTENTION_DURATION['C']
            )

        # 添加到记忆
        self.memory.add_observation(
            exhibit_id=exhibit_id,
            exhibit_name=exhibit_name,
            attention_level=attention_level,
            estimated_duration=estimated_duration,
            timestamp=timestamp
        )

        # 返回最新记录
        return self.memory.get_recent(1)[-1]

    def predict_next(self, verbose: bool = True) -> Dict:
        """
        预测下一个目标

        Args:
            verbose: 是否打印详细信息

        Returns:
            预测结果字典
        """
        # 获取当前和历史
        recent = self.memory.get_recent(1)
        if not recent:
            return {
                "error": "No history yet. Please add at least one observation.",
                "prediction_id": None
            }

        current = recent[-1]
        history = self.memory.get_recent()

        # 构建上下文
        context = self.context_builder.build_context(current, history)

        # 预测
        if verbose:
            return self.reasoner.predict_with_explanation(
                context,
                self.config.attention,
                verbose=True
            )
        else:
            return self.reasoner.predict_next(context, self.config.attention)

    def predict_sequence(
        self,
        n_steps: int = 5,
        exclude_visited: bool = True,
        verbose: bool = True
    ) -> List[Dict]:
        """
        预测未来序列

        Args:
            n_steps: 预测步数 (默认5)
            exclude_visited: 是否排除已访问的展品 (默认True)
            verbose: 是否打印详细信息

        Returns:
            预测序列列表
        """
        # 获取当前和历史
        recent = self.memory.get_recent(1)
        if not recent:
            print("No history yet. Please add at least one observation.")
            return []

        current = recent[-1]
        history = self.memory.get_recent()

        # 构建上下文
        context = self.context_builder.build_context(current, history)

        # 预测序列
        sequence = self.sequence_predictor.predict_sequence(
            context,
            n_steps=n_steps,
            exclude_visited=exclude_visited
        )

        if verbose:
            print(self.sequence_predictor.format_sequence_summary(sequence))

        return sequence

    def get_memory_statistics(self) -> Dict:
        """
        获取记忆统计信息

        Returns:
            统计信息字典
        """
        return self.memory.get_statistics()

    def get_recent_history(self, n: int = 5) -> List[GazeRecord]:
        """
        获取最近的历史记录

        Args:
            n: 记录数量

        Returns:
            GazeRecord列表
        """
        return self.memory.get_recent(n)

    def get_visit_count(self, exhibit_id: str) -> int:
        """
        获取特定展品的访问次数

        Args:
            exhibit_id: 展品ID

        Returns:
            访问次数
        """
        return self.memory.get_visit_count(exhibit_id)

    def has_visited(self, exhibit_id: str) -> bool:
        """
        检查是否访问过某展品

        Args:
            exhibit_id: 展品ID

        Returns:
            是否访问过
        """
        return self.memory.has_visited(exhibit_id)

    def clear_memory(self):
        """清空记忆"""
        self.memory.clear()
        print("Memory cleared.")

    def get_reachable_exhibits(self, exhibit_id: str) -> List[Dict]:
        """
        获取从指定展品可直接到达的所有展品

        Args:
            exhibit_id: 展品ID

        Returns:
            可达展品列表
        """
        return self.context_builder.get_reachable_exhibits(exhibit_id)

    def simulate_trajectory(
        self,
        exhibit_ids: List[str],
        attention_levels: List[str]
    ):
        """
        模拟一段轨迹（批量添加观测）

        Args:
            exhibit_ids: 展品ID列表
            attention_levels: 注意力等级列表（与exhibit_ids一一对应）
        """
        if len(exhibit_ids) != len(attention_levels):
            raise ValueError("exhibit_ids and attention_levels must have same length")

        for eid, level in zip(exhibit_ids, attention_levels):
            # 从拓扑获取展品名称
            node_data = self.topology.query_node(eid)
            if "error" in node_data:
                print(f"Warning: Exhibit {eid} not found in topology")
                name = f"Exhibit {eid}"
            else:
                name = node_data['info']['name']

            # 添加观测
            self.add_observation(eid, name, level)

        print(f"Added {len(exhibit_ids)} observations to memory.")

    def export_predictions(
        self,
        sequence: List[Dict],
        output_path: str = None
    ) -> str:
        """
        导出预测结果为JSON格式

        Args:
            sequence: 预测序列列表
            output_path: 输出文件路径，如果为None则自动生成

        Returns:
            输出文件路径
        """
        import json
        from datetime import datetime

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/outputs/predictions/prediction_{timestamp}.json"

        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 准备导出数据
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'map_name': self.map_name,
            'memory_statistics': self.get_memory_statistics(),
            'predictions': sequence
        }

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"Predictions exported to: {output_path}")
        return output_path

    def __len__(self) -> int:
        """返回长期记忆中的记录总数"""
        return len(self.memory)

    def __repr__(self) -> str:
        return (f"PredictionEngine(map={self.map_name}, "
                f"memory={self.memory}, "
                f"topology_nodes={self.topology.graph.number_of_nodes()})")
