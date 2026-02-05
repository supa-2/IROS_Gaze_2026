"""
Memory Management Module

This module provides memory management for the Eye-LLM system including:
- ShortTermMemory: Sliding window buffer for recent gaze records
- LongTermMemory: Persistent storage with statistics for all historical records
- MemoryManager: Unified interface coordinating both memory systems
- GazeRecord: Data structure for individual gaze observations
"""

from .manager import MemoryManager, GazeRecord
from .short_term import ShortTermMemory
from .long_term import LongTermMemory

__all__ = [
    'MemoryManager',
    'GazeRecord',
    'ShortTermMemory',
    'LongTermMemory'
]
