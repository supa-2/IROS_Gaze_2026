"""
全局配置文件 - Eye-LLM Spatial Intent Prediction System

This module contains all configuration classes for the system including:
- Model configurations (LLM, VLM, SAM2)
- Memory configurations (short-term and long-term)
- Attention level configurations (duration mappings)
"""

from dataclasses import dataclass
from typing import Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """模型配置"""

    # LLM配置
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    llm_api_key: str = None
    llm_base_url: str = None

    # VLM配置 (GPT-4V已整合到GPT-4o)
    vlm_model: str = "gpt-4o"
    vlm_max_image_size: int = 20 * 1024 * 1024  # 20MB

    # SAM 2 API配置（Replicate）
    sam2_api_provider: str = "replicate"
    replicate_api_token: str = None
    sam2_model_version: str = "lucataco/segment-anything-2"  # 可用模型（免费账户有速率限制）

    def __post_init__(self):
        """从环境变量加载配置"""
        if self.llm_api_key is None:
            self.llm_api_key = os.getenv("OPENAI_API_KEY", "")
        if self.llm_base_url is None:
            self.llm_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if self.replicate_api_token is None:
            self.replicate_api_token = os.getenv("REPLICATE_API_TOKEN", "")

        # 从环境变量覆盖模型配置
        if os.getenv("LLM_MODEL"):
            self.llm_model = os.getenv("LLM_MODEL")
        if os.getenv("VLM_MODEL"):
            self.vlm_model = os.getenv("VLM_MODEL")
        if os.getenv("SAM2_MODEL_VERSION"):
            self.sam2_model_version = os.getenv("SAM2_MODEL_VERSION")


@dataclass
class MemoryConfig:
    """记忆配置"""

    short_term_size: int = 5
    long_term_max_size: int = 10000

    def __post_init__(self):
        """从环境变量加载配置"""
        if os.getenv("SHORT_TERM_SIZE"):
            self.short_term_size = int(os.getenv("SHORT_TERM_SIZE"))
        if os.getenv("LONG_TERM_MAX_SIZE"):
            self.long_term_max_size = int(os.getenv("LONG_TERM_MAX_SIZE"))


@dataclass
class AttentionConfig:
    """注意力等级配置"""

    # 指数递增: A=120s, B=60s, C=30s, D=15s, E=5s
    ATTENTION_DURATION: Dict[str, int] = None
    ATTENTION_DESCRIPTION: Dict[str, str] = None

    def __post_init__(self):
        if self.ATTENTION_DURATION is None:
            self.ATTENTION_DURATION = {
                'A': int(os.getenv("ATTENTION_A", "120")),
                'B': int(os.getenv("ATTENTION_B", "60")),
                'C': int(os.getenv("ATTENTION_C", "30")),
                'D': int(os.getenv("ATTENTION_D", "15")),
                'E': int(os.getenv("ATTENTION_E", "5"))
            }

        if self.ATTENTION_DESCRIPTION is None:
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

    model: ModelConfig = None
    memory: MemoryConfig = None
    attention: AttentionConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.attention is None:
            self.attention = AttentionConfig()

    def validate(self) -> bool:
        """验证配置是否完整"""
        errors = []

        # 验证必需的API密钥
        if not self.model.llm_api_key:
            errors.append("Missing OpenAI API Key (OPENAI_API_KEY)")

        # 验证注意力等级
        valid_levels = {'A', 'B', 'C', 'D', 'E'}
        for level in self.attention.ATTENTION_DURATION.keys():
            if level not in valid_levels:
                errors.append(f"Invalid attention level: {level}")

        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True


# 全局单例
config = Config()
