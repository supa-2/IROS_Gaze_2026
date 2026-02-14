"""
Gaze Records - 数据类定义

This module defines the data classes used for storing gaze records.
Separated to avoid circular import issues.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class GazeRecord:
    """
    单次凝视记录 (Gaze Fixation Record)

    Attributes:
        exhibit_id: 展品ID (e.g., 'TH-E01', 'TH-B02')
        exhibit_name: 展品名称 (e.g., '欢迎致辞', '何尊')
        timestamp: 观测时间戳
        attention_level: 注意力等级 (A/B/C/D/E)
        estimated_duration: 预计停留时间(秒)
        actual_duration: 实际停留时间(可选, 秒)
    """
    exhibit_id: str
    exhibit_name: str
    timestamp: datetime
    attention_level: str
    estimated_duration: int
    actual_duration: Optional[int] = None

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'exhibit_id': self.exhibit_id,
            'exhibit_name': self.exhibit_name,
            'timestamp': self.timestamp.isoformat(),
            'attention_level': self.attention_level,
            'estimated_duration': self.estimated_duration,
            'actual_duration': self.actual_duration
        }
