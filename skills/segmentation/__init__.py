"""
Semantic Segmentation Module

This module provides semantic segmentation capabilities using SAM 2 via Replicate API:
- ReplicateSAM2Wrapper: Cloud-based SAM 2 API integration
- MaskProcessor: Mask filtering and NMS
- CenterExtractor: Centroid computation
- SemanticSegmenter: Main orchestrator
"""

from .segmenter import SemanticSegmenter

__all__ = ['SemanticSegmenter']
