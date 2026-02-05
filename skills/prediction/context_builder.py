"""
Context Builder - 上下文构建器

This module builds comprehensive prediction contexts by fusing memory data
with spatial topology information. It transforms raw gaze records and
graph data into structured prompts for LLM reasoning.
"""

from typing import Dict, List
import sys
import os

# Add topology to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'topology'))
from graph_engine import TopologyEngine

from ..memory.manager import GazeRecord


class ContextBuilder:
    """
    上下文构建器 - 融合记忆和环境信息

    This class is responsible for building comprehensive prediction contexts
    by combining:
    1. Current gaze information
    2. Historical behavior patterns
    3. Spatial topology constraints
    4. Exhibition hall features
    """

    def __init__(self, topology: TopologyEngine):
        """
        初始化上下文构建器

        Args:
            topology: TopologyEngine实例，提供空间拓扑信息
        """
        self.topology = topology

    def build_context(
        self,
        current_gaze: GazeRecord,
        history: List[GazeRecord]
    ) -> Dict:
        """
        构建完整的预测上下文

        Args:
            current_gaze: 当前凝视记录
            history: 历史凝视记录列表

        Returns:
            包含以下键的字典:
            - current: 当前位置信息 (id, name, features, attention_level)
            - history: 历史行为摘要
            - spatial: 空间环境信息 (previous_path, next_path, reachable_options)
            - statistics: 统计数据 (total_gazes, unique_exhibits)
        """
        # 1. 从拓扑引擎获取空间信息
        spatial_context = self.topology.query_node(current_gaze.exhibit_id)

        if "error" in spatial_context:
            return {
                "error": spatial_context["error"],
                "current": {
                    "id": current_gaze.exhibit_id,
                    "name": current_gaze.exhibit_name,
                    "attention_level": current_gaze.attention_level
                },
                "history": self._summarize_history(history),
                "spatial": {},
                "statistics": {
                    "total_gazes": len(history),
                    "unique_exhibits": len(set(g.exhibit_id for g in history))
                }
            }

        # 2. 获取历史行为序列
        history_summary = self._summarize_history(history)

        # 3. 获取当前展品的详细特征
        current_info = spatial_context.get('info', {})

        # 4. 获取可达选项
        reachable = spatial_context.get('context', {}).get('direct_choices', [])

        return {
            'current': {
                'id': current_gaze.exhibit_id,
                'name': current_gaze.exhibit_name,
                'features': current_info.get('features', ''),
                'attention_level': current_gaze.attention_level,
                'estimated_duration': current_gaze.estimated_duration
            },
            'history': history_summary,
            'spatial': {
                'previous_path': spatial_context.get('context', {}).get('previous_path', []),
                'next_path': spatial_context.get('context', {}).get('next_path', []),
                'reachable_options': reachable
            },
            'statistics': {
                'total_gazes': len(history),
                'unique_exhibits': len(set(g.exhibit_id for g in history)),
                'visited_exhibits': list(set(g.exhibit_id for g in history))
            }
        }

    def _summarize_history(self, history: List[GazeRecord]) -> List[Dict]:
        """
        总结历史行为

        Args:
            history: GazeRecord列表

        Returns:
            摘要字典列表，每个包含 id, name, level, duration
        """
        return [
            {
                'id': g.exhibit_id,
                'name': g.exhibit_name,
                'level': g.attention_level,
                'duration': g.estimated_duration
            }
            for g in history[-5:]  # 最近5个
        ]

    def format_context_for_llm(self, context: Dict) -> str:
        """
        将上下文格式化为LLM友好的文本

        Args:
            context: build_context()返回的上下文字典

        Returns:
            格式化的字符串，可直接用于LLM prompt
        """
        if "error" in context:
            return f"Error: {context['error']}"

        current = context['current']
        history = context['history']
        spatial = context['spatial']

        # 构建当前状态部分
        prompt = f"""**当前状态**
- 位置: {current['name']} ({current['id']})
- 特征: {current['features']}
- 当前注意力: {current['attention_level']} (预计停留{current['estimated_duration']}秒)

**历史行为** (最近{len(history)}个)
"""

        # 添加历史记录
        for i, h in enumerate(history, 1):
            prompt += f"{i}. [{h['level']}] {h['name']} - {h['duration']}s\n"

        # 添加空间环境部分
        prompt += "\n**空间环境**\n"

        # 前序路径
        prev_path = spatial.get('previous_path', [])
        if prev_path:
            prev_str = " -> ".join([f"{n['name']}({n['id']})" for n in prev_path])
            prompt += f"- 动线前序: {prev_str}\n"

        # 后续路径
        next_path = spatial.get('next_path', [])
        if next_path:
            next_str = " -> ".join([f"{n['name']}({n['id']})" for n in next_path])
            prompt += f"- 动线后续: {next_str}\n"

        # 可达选项
        prompt += "- 可达选项:\n"
        for opt in spatial.get('reachable_options', []):
            # 获取邻居节点的详细信息
            neighbor_info = self.topology.graph.nodes.get(opt['id'], {})
            neighbor_name = neighbor_info.get('name', 'Unknown')
            prompt += f"  • {opt['id']} ({neighbor_name}) - 关系: {opt['relation']}\n"

        # 统计信息
        stats = context['statistics']
        prompt += f"\n**统计信息**\n"
        prompt += f"- 总凝视次数: {stats['total_gazes']}\n"
        prompt += f"- 已访问展品数: {stats['unique_exhibits']}\n"

        return prompt

    def get_reachable_exhibits(self, exhibit_id: str) -> List[Dict]:
        """
        获取从指定展品可直接到达的所有展品

        Args:
            exhibit_id: 展品ID

        Returns:
            可达展品列表，每个包含 id, name, relation
        """
        context_data = self.topology.query_node(exhibit_id)

        if "error" in context_data:
            return []

        choices = context_data.get('context', {}).get('direct_choices', [])

        # 增强信息，添加展品名称
        enriched_choices = []
        for choice in choices:
            node_info = self.topology.graph.nodes.get(choice['id'], {})
            enriched_choices.append({
                'id': choice['id'],
                'name': node_info.get('name', 'Unknown'),
                'relation': choice['relation'],
                'features': node_info.get('features', '')
            })

        return enriched_choices
