"""
Sequence Predictor - 序列预测器

This module implements multi-step sequence prediction for forecasting
future viewing trajectories beyond just the next gaze target.
"""

from typing import List, Dict
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'topology'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from graph_engine import TopologyEngine
from .llm_reasoner import LLMReasoner
from ..memory.manager import GazeRecord, MemoryManager
from config import AttentionConfig


class SequencePredictor:
    """
    序列预测器 - 预测未来N步

    This class predicts multi-step future viewing sequences by iteratively
    calling the LLM reasoner and updating the prediction context.
    """

    def __init__(
        self,
        reasoner: LLMReasoner,
        topology: TopologyEngine,
        attention_config: AttentionConfig
    ):
        """
        初始化序列预测器

        Args:
            reasoner: LLMReasoner实例
            topology: TopologyEngine实例
            attention_config: AttentionConfig实例
        """
        self.reasoner = reasoner
        self.topology = topology
        self.attention_config = attention_config

    def predict_sequence(
        self,
        current_context: Dict,
        n_steps: int = 5,
        exclude_visited: bool = True
    ) -> List[Dict]:
        """
        预测未来n步的观看序列

        Args:
            current_context: 当前上下文字典 (from ContextBuilder)
            n_steps: 预测步数 (默认5)
            exclude_visited: 是否排除已访问的展品 (默认True)

        Returns:
            预测序列列表，每个元素包含:
            - prediction_id: 展品ID
            - prediction_name: 展品名称
            - attention_level: 注意力等级
            - estimated_duration: 预计停留时间
            - confidence: 置信度
            - reasoning: 推理过程
            - step_number: 步数
        """
        sequence = []

        # 复制上下文以避免修改原始数据
        context = current_context.copy()
        visited_ids = set(current_context.get('statistics', {}).get('visited_exhibits', []))

        for step in range(1, n_steps + 1):
            # 预测下一步
            prediction = self.reasoner.predict_next(context, self.attention_config)

            # 检查错误
            if 'error' in prediction:
                print(f"Step {step}: Prediction failed - {prediction['error']}")
                break

            prediction_id = prediction.get('prediction_id')

            # 检查是否应该停止
            if not prediction_id:
                print(f"Step {step}: No valid prediction")
                break

            # 检查是否重复访问
            if exclude_visited and prediction_id in visited_ids:
                print(f"Step {step}: Predicted already visited exhibit {prediction_id}, stopping")
                break

            # 添加步数
            prediction['step_number'] = step
            sequence.append(prediction)

            # 更新已访问集合
            visited_ids.add(prediction_id)

            # 更新上下文（模拟移动到预测点）
            context = self._update_context_for_next_step(
                context,
                prediction,
                visited_ids
            )

        return sequence

    def _update_context_for_next_step(
        self,
        current_context: Dict,
        prediction: Dict,
        visited_ids: set
    ) -> Dict:
        """
        更新上下文以进行下一步预测

        Args:
            current_context: 当前上下文
            prediction: 预测结果
            visited_ids: 已访问的展品ID集合

        Returns:
            更新后的上下文字典
        """
        prediction_id = prediction['prediction_id']
        prediction_name = prediction['prediction_name']
        attention_level = prediction['attention_level']
        estimated_duration = prediction['estimated_duration']

        # 查询新位置的空间信息
        spatial_data = self.topology.query_node(prediction_id)

        # 更新历史
        updated_history = current_context.get('history', []) + [{
            'id': prediction_id,
            'name': prediction_name,
            'level': attention_level,
            'duration': estimated_duration
        }]

        # 只保留最近5个
        updated_history = updated_history[-5:]

        # 构建新上下文
        new_context = {
            'current': {
                'id': prediction_id,
                'name': prediction_name,
                'features': spatial_data.get('info', {}).get('features', ''),
                'attention_level': attention_level,
                'estimated_duration': estimated_duration
            },
            'history': updated_history,
            'spatial': {
                'previous_path': spatial_data.get('context', {}).get('previous_path', []),
                'next_path': spatial_data.get('context', {}).get('next_path', []),
                'reachable_options': spatial_data.get('context', {}).get('direct_choices', [])
            },
            'statistics': {
                'total_gazes': current_context['statistics']['total_gazes'] + 1,
                'unique_exhibits': len(visited_ids),
                'visited_exhibits': list(visited_ids)
            }
        }

        return new_context

    def predict_with_alternatives(
        self,
        current_context: Dict,
        n_steps: int = 3,
        n_alternatives: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        预测多条可能的路径

        Args:
            current_context: 当前上下文
            n_steps: 每条路径的步数
            n_alternatives: 替代路径数量

        Returns:
            字典，键为路径名称 ('primary', 'alternative_1', etc.)，
            值为预测序列列表
        """
        paths = {}

        # 主路径
        primary_path = self.predict_sequence(current_context, n_steps)
        paths['primary'] = primary_path

        # 替代路径 (简化实现：基于第一跳的不同选择)
        reachable = current_context.get('spatial', {}).get('reachable_options', [])

        for i, alt_id in enumerate(reachable[:n_alternatives]):
            # 创建替代起点
            alt_context = self._create_alternative_context(
                current_context,
                alt_id['id']
            )

            # 预测替代路径
            alt_path = self.predict_sequence(alt_context, n_steps - 1)
            paths[f'alternative_{i+1}'] = alt_path

        return paths

    def _create_alternative_context(
        self,
        original_context: Dict,
        first_step_id: str
    ) -> Dict:
        """
        创建替代路径的起始上下文

        Args:
            original_context: 原始上下文
            first_step_id: 第一步的展品ID

        Returns:
            新的上下文字典
        """
        # 查询第一步的空间信息
        spatial_data = self.topology.query_node(first_step_id)
        node_info = spatial_data.get('info', {})

        # 默认注意力等级和持续时间
        default_attention = 'C'
        default_duration = self.attention_config.ATTENTION_DURATION[default_attention]

        new_context = {
            'current': {
                'id': first_step_id,
                'name': node_info.get('name', 'Unknown'),
                'features': node_info.get('features', ''),
                'attention_level': default_attention,
                'estimated_duration': default_duration
            },
            'history': original_context.get('history', []),
            'spatial': {
                'previous_path': spatial_data.get('context', {}).get('previous_path', []),
                'next_path': spatial_data.get('context', {}).get('next_path', []),
                'reachable_options': spatial_data.get('context', {}).get('direct_choices', [])
            },
            'statistics': original_context.get('statistics', {})
        }

        return new_context

    def format_sequence_summary(self, sequence: List[Dict]) -> str:
        """
        格式化预测序列为易读的字符串

        Args:
            sequence: 预测序列列表

        Returns:
            格式化的字符串
        """
        if not sequence:
            return "No predictions generated."

        lines = [
            "="*60,
            "🔮 Predicted Viewing Sequence",
            "="*60
        ]

        for pred in sequence:
            step = pred['step_number']
            name = pred['prediction_name']
            aid = pred['prediction_id']
            level = pred['attention_level']
            duration = pred['estimated_duration']
            conf = pred.get('confidence', 0.0)

            lines.append(
                f"{step}. [{level}] {name} ({aid}) - {duration}s "
                f"(confidence: {conf:.2f})"
            )

        # 计算总时间
        total_duration = sum(p['estimated_duration'] for p in sequence)
        lines.append("="*60)
        lines.append(f"Total Estimated Duration: {total_duration}s ({total_duration/60:.1f} minutes)")
        lines.append("="*60)

        return "\n".join(lines)
