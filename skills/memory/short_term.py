"""
Short-Term Memory - 短期记忆系统

This module implements a sliding window buffer for storing recent gaze records.
It maintains the most recent observations and automatically discards old records
when the buffer is full.
"""

from collections import deque
from typing import List

from .records import GazeRecord


class ShortTermMemory:
    """
    短期记忆 - 滑动窗口实现

    This class provides a fixed-size sliding window buffer for storing
    recent gaze records. When the buffer is full, adding new records
    automatically removes the oldest records (FIFO).

    Attributes:
        max_size: 缓冲区最大容量
        buffer: 双端队列，存储GazeRecord对象
    """

    def __init__(self, max_size: int = 5):
        """
        初始化短期记忆

        Args:
            max_size: 短期记忆容量 (默认5条记录)
        """
        self.max_size = max_size
        self.buffer: deque[GazeRecord] = deque(maxlen=max_size)

    def add(self, record: GazeRecord):
        """
        添加凝视记录

        如果缓冲区已满，最旧的记录会被自动移除。

        Args:
            record: GazeRecord对象
        """
        self.buffer.append(record)

    def get_recent(self, n: int = None) -> List[GazeRecord]:
        """
        获取最近n条记录

        Args:
            n: 获取的记录数量，如果为None则返回所有记录

        Returns:
            最近n条GazeRecord列表，按时间从旧到新排序
        """
        if n is None:
            return list(self.buffer)

        n = min(n, len(self.buffer))
        return list(self.buffer)[-n:]

    def get_latest(self) -> GazeRecord:
        """
        获取最新的一条记录

        Returns:
            最新的GazeRecord对象，如果缓冲区为空则返回None
        """
        return self.buffer[-1] if self.buffer else None

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()

    def is_empty(self) -> bool:
        """
        检查缓冲区是否为空

        Returns:
            是否为空
        """
        return len(self.buffer) == 0

    def is_full(self) -> bool:
        """
        检查缓冲区是否已满

        Returns:
            是否已满
        """
        return len(self.buffer) == self.max_size

    def get_attention_sequence(self) -> List[str]:
        """
        获取最近的注意力等级序列

        Returns:
            注意力等级列表，如 ['A', 'B', 'C']
        """
        return [record.attention_level for record in self.buffer]

    def get_exhibit_sequence(self) -> List[str]:
        """
        获取最近的展品ID序列

        Returns:
            展品ID列表，如 ['TH-E01', 'TH-B02', 'TH-I-B01']
        """
        return [record.exhibit_id for record in self.buffer]

    def __len__(self) -> int:
        """返回当前缓冲区中的记录数量"""
        return len(self.buffer)

    def __repr__(self) -> str:
        return (f"ShortTermMemory(size={len(self.buffer)}/{self.max_size}, "
                f"latest={self.buffer[-1].exhibit_id if self.buffer else 'None'})")
