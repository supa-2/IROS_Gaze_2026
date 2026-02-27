#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiments Package - 实验包

包含消融实验、对照实验和评估工具
"""

from .feature_extractor import FeatureExtractor, batch_extract_features
from .ablation_study import AblationConfig, AblationPredictor, AblationExperiment
from .baseline_comparison import (
    MarkovBaseline,
    LSTMBaseline,
    ZeroShotLLMBaseline,
    OurMethod,
    BaselineComparison
)

__all__ = [
    'FeatureExtractor',
    'batch_extract_features',
    'AblationConfig',
    'AblationPredictor',
    'AblationExperiment',
    'MarkovBaseline',
    'LSTMBaseline',
    'ZeroShotLLMBaseline',
    'OurMethod',
    'BaselineComparison'
]
