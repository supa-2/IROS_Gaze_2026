"""
全局配置文件 - Eye-LLM Spatial Intent Prediction System
"""

from dataclasses import dataclass, field
from typing import Dict
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """模型配置"""
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    llm_api_key: str = None
    llm_base_url: str = None
    vlm_model: str = "gpt-4o"
    vlm_max_image_size: int = 20 * 1024 * 1024
    sam2_api_provider: str = "replicate"
    replicate_api_token: str = None
    sam2_model_version: str = "meta/segment-anything-2"

    def __post_init__(self):
        if self.llm_api_key is None:
            self.llm_api_key = os.getenv("OPENAI_API_KEY", "")
        if self.llm_base_url is None:
            self.llm_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if self.replicate_api_token is None:
            self.replicate_api_token = os.getenv("REPLICATE_API_TOKEN", "")


@dataclass
class MemoryConfig:
    """记忆配置"""
    short_term_size: int = 5
    long_term_max_size: int = 10000

    def __post_init__(self):
        if os.getenv("SHORT_TERM_SIZE"):
            self.short_term_size = int(os.getenv("SHORT_TERM_SIZE"))
        if os.getenv("LONG_TERM_MAX_SIZE"):
            self.long_term_max_size = int(os.getenv("LONG_TERM_MAX_SIZE"))


@dataclass
class AttentionConfig:
    """注意力等级配置 - 指数递增: A=120s, B=60s, C=30s, D=15s, E=5s"""
    ATTENTION_DURATION: Dict[str, int] = field(default_factory=dict)
    ATTENTION_DESCRIPTION: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.ATTENTION_DURATION:
            self.ATTENTION_DURATION = {
                'A': int(os.getenv("ATTENTION_A", "120")),
                'B': int(os.getenv("ATTENTION_B", "60")),
                'C': int(os.getenv("ATTENTION_C", "30")),
                'D': int(os.getenv("ATTENTION_D", "15")),
                'E': int(os.getenv("ATTENTION_E", "5"))
            }
        if not self.ATTENTION_DESCRIPTION:
            self.ATTENTION_DESCRIPTION = {
                'A': '深度关注 - 长时间仔细观看',
                'B': '中等关注 - 正常观看',
                'C': '一般关注 - 浏览式观看',
                'D': '快速浏览 - 短暂停留',
                'E': '一瞥而过 - 快速扫视'
            }


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)


# 全局单例
config = Config()
