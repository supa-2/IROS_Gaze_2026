"""
Long-Term Memory - 长期记忆系统

This module implements persistent storage for all historical gaze records
with statistics tracking including visit counts, duration statistics, and
frequency analysis.
"""

from typing import List, Dict, Tuple
from collections import defaultdict

from .manager import GazeRecord


class LongTermMemory:
    """
    长期记忆 - 统计分析实现

    This class provides persistent storage for all historical gaze records.
    It tracks visit frequencies, dwell times, and provides various
    statistical analyses of user behavior patterns.

    Attributes:
        max_size: 最大记录容量
        records: 所有历史记录列表
        visit_count: 每个展品的访问频率统计
        total_durations: 每个展品的总停留时间统计
    """

    def __init__(self, max_size: int = 10000):
        """
        初始化长期记忆

        Args:
            max_size: 最大记录容量 (默认10000条)
        """
        self.max_size = max_size
        self.records: List[GazeRecord] = []
        self.visit_count: Dict[str, int] = defaultdict(int)
        self.total_durations: Dict[str, int] = defaultdict(int)

    def add(self, record: GazeRecord):
        """
        添加凝视记录

        Args:
            record: GazeRecord对象
        """
        self.records.append(record)
        self.visit_count[record.exhibit_id] += 1

        # 统计停留时间（使用实际时间或估计时间）
        duration = record.actual_duration if record.actual_duration else record.estimated_duration
        self.total_durations[record.exhibit_id] += duration

        # 如果超过最大容量，移除最旧的记录
        if len(self.records) > self.max_size:
            oldest = self.records.pop(0)
            self.visit_count[oldest.exhibit_id] -= 1
            if self.visit_count[oldest.exhibit_id] == 0:
                del self.visit_count[oldest.exhibit_id]

            duration = oldest.actual_duration if oldest.actual_duration else oldest.estimated_duration
            self.total_durations[oldest.exhibit_id] -= duration
            if self.total_durations[oldest.exhibit_id] == 0:
                del self.total_durations[oldest.exhibit_id]

    def get_all(self) -> List[GazeRecord]:
        """
        获取所有历史记录

        Returns:
            所有GazeRecord列表，按时间从旧到新排序
        """
        return list(self.records)

    def unique_count(self) -> int:
        """
        获取唯一展品数量

        Returns:
            访问过的不同展品数量
        """
        return len(self.visit_count)

    def most_visited(self, n: int = 5) -> List[Tuple[str, int]]:
        """
        返回访问最多的前n个展品

        Args:
            n: 返回的展品数量 (默认5个)

        Returns:
            (展品ID, 访问次数) 元组列表，按访问次数降序排序
        """
        return sorted(
            self.visit_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

    def get_visit_count(self, exhibit_id: str) -> int:
        """
        获取特定展品的访问次数

        Args:
            exhibit_id: 展品ID

        Returns:
            访问次数，如果未访问过则返回0
        """
        return self.visit_count.get(exhibit_id, 0)

    def has_visited(self, exhibit_id: str) -> bool:
        """
        检查是否访问过某展品

        Args:
            exhibit_id: 展品ID

        Returns:
            是否访问过
        """
        return exhibit_id in self.visit_count

    def get_exhibits_by_attention(self, attention_level: str) -> List[str]:
        """
        获取特定注意力等级的所有展品ID

        Args:
            attention_level: 注意力等级 (A/B/C/D/E)

        Returns:
            展品ID列表
        """
        return [
            record.exhibit_id
            for record in self.records
            if record.attention_level == attention_level
        ]

    def total_duration(self) -> int:
        """
        计算总停留时间

        Returns:
            所有展品的总停留时间(秒)
        """
        return sum(self.total_durations.values())

    def average_duration(self) -> float:
        """
        计算平均停留时间

        Returns:
            每次访问的平均停留时间(秒)
        """
        total_visits = sum(self.visit_count.values())
        if total_visits == 0:
            return 0.0
        return self.total_duration() / total_visits

    def get_exhibit_duration(self, exhibit_id: str) -> int:
        """
        获取特定展品的总停留时间

        Args:
            exhibit_id: 展品ID

        Returns:
            总停留时间(秒)，如果未访问过则返回0
        """
        return self.total_durations.get(exhibit_id, 0)

    def get_average_duration_per_exhibit(self, exhibit_id: str) -> float:
        """
        获取特定展品的平均停留时间

        Args:
            exhibit_id: 展品ID

        Returns:
            平均停留时间(秒)，如果未访问过则返回0
        """
        visits = self.get_visit_count(exhibit_id)
        if visits == 0:
            return 0.0
        return self.total_durations[exhibit_id] / visits

    def get_attention_distribution(self) -> Dict[str, int]:
        """
        获取注意力等级分布统计

        Returns:
            各注意力等级的次数，如 {'A': 5, 'B': 10, 'C': 8}
        """
        distribution = defaultdict(int)
        for record in self.records:
            distribution[record.attention_level] += 1
        return dict(distribution)

    def clear(self):
        """清空所有记录"""
        self.records.clear()
        self.visit_count.clear()
        self.total_durations.clear()

    def __len__(self) -> int:
        """返回记录总数"""
        return len(self.records)

    def __repr__(self) -> str:
        return (f"LongTermMemory(records={len(self.records)}, "
                f"unique={self.unique_count()}, "
                f"total_duration={self.total_duration()}s)")
