"""
Memory Manager - 整合短期和长期记忆系统

This module provides the main MemoryManager class that coordinates
short-term and long-term memory systems for tracking gaze behavior.
"""

from datetime import datetime
from typing import List, Dict, Optional

from .records import GazeRecord
from .short_term import ShortTermMemory
from .long_term import LongTermMemory


class MemoryManager:
    """
    记忆管理器 - 整合短期和长期记忆

    This class provides a unified interface for managing both short-term
    and long-term memory systems. It automatically distributes new gaze
    records to both memory systems and provides combined statistics.
    """

    def __init__(self, short_term_size: int = 5, long_term_max_size: int = 10000):
        """
        初始化记忆管理器

        Args:
            short_term_size: 短期记忆容量 (默认5条)
            long_term_max_size: 长期记忆最大容量 (默认10000条)
        """
        self.short_term = ShortTermMemory(max_size=short_term_size)
        self.long_term = LongTermMemory(max_size=long_term_max_size)

    def add_gaze(self, record: GazeRecord):
        """
        添加凝视记录到记忆系统

        新记录会同时添加到短期记忆和长期记忆中。

        Args:
            record: GazeRecord对象
        """
        self.short_term.add(record)
        self.long_term.add(record)

    def add_observation(
        self,
        exhibit_id: str,
        exhibit_name: str,
        attention_level: str,
        estimated_duration: int,
        timestamp: Optional[datetime] = None
    ):
        """
        便捷方法：直接添加观测数据

        Args:
            exhibit_id: 展品ID
            exhibit_name: 展品名称
            attention_level: 注意力等级 (A/B/C/D/E)
            estimated_duration: 预计停留时间(秒)
            timestamp: 时间戳 (默认为当前时间)
        """
        if timestamp is None:
            timestamp = datetime.now()

        record = GazeRecord(
            exhibit_id=exhibit_id,
            exhibit_name=exhibit_name,
            timestamp=timestamp,
            attention_level=attention_level,
            estimated_duration=estimated_duration
        )

        self.add_gaze(record)

    def get_recent(self, n: int = 5) -> List[GazeRecord]:
        """
        获取最近n条记录

        Args:
            n: 获取的记录数量 (默认5条)

        Returns:
            最近n条GazeRecord列表
        """
        return self.short_term.get_recent(n)

    def get_all_history(self) -> List[GazeRecord]:
        """
        获取所有历史记录

        Returns:
            所有GazeRecord列表
        """
        return self.long_term.get_all()

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        Returns:
            包含以下键的字典:
            - short_term_count: 短期记忆记录数
            - long_term_count: 长期记忆记录数
            - unique_exhibits: 唯一展品数量
            - most_visited: 访问最多的前5个展品
            - total_duration: 总停留时间(秒)
            - average_duration: 平均停留时间(秒)
        """
        return {
            'short_term_count': len(self.short_term),
            'long_term_count': len(self.long_term),
            'unique_exhibits': self.long_term.unique_count(),
            'most_visited': self.long_term.most_visited(5),
            'total_duration': self.long_term.total_duration(),
            'average_duration': self.long_term.average_duration()
        }

    def get_visit_count(self, exhibit_id: str) -> int:
        """
        获取特定展品的访问次数

        Args:
            exhibit_id: 展品ID

        Returns:
            访问次数
        """
        return self.long_term.get_visit_count(exhibit_id)

    def has_visited(self, exhibit_id: str) -> bool:
        """
        检查是否访问过某展品

        Args:
            exhibit_id: 展品ID

        Returns:
            是否访问过
        """
        return self.long_term.has_visited(exhibit_id)

    def clear(self):
        """清空所有记忆"""
        self.short_term.clear()
        self.long_term.clear()

    def __len__(self) -> int:
        """返回长期记忆中的记录总数"""
        return len(self.long_term)

    def __repr__(self) -> str:
        return (f"MemoryManager(short_term={len(self.short_term)}, "
                f"long_term={len(self.long_term)}, "
                f"unique={self.long_term.unique_count()})")
