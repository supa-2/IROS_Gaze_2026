"""
Semantic Segmenter - 语义分割器

This module provides semantic segmentation using SAM 2 (Segment Anything Model 2)
via Replicate API, eliminating the need for local GPU resources.
"""

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import replicate

from config import ModelConfig


class SemanticSegmenter:
    """
    语义分割器 - 整合分割流程

    This class provides a complete segmentation pipeline using SAM 2 via
    Replicate API. It handles image loading, segmentation, mask processing,
    and result preparation for downstream VLM analysis.
    """

    def __init__(self, config: ModelConfig):
        """
        初始化语义分割器

        Args:
            config: ModelConfig对象，包含SAM 2配置
        """
        self.config = config
        self.client = None

        # 延迟初始化Replicate客户端（只有在需要时才连接）
        if config.replicate_api_token:
            os.environ['REPLICATE_API_TOKEN'] = config.replicate_api_token
            try:
                self.client = replicate.Client(api_token=config.replicate_api_token)
            except Exception as e:
                print(f"Warning: Failed to initialize Replicate client: {e}")
                print("Segmentation features will be disabled.")

    def segment_image(
        self,
        image_path: str,
        mask_prompt: str = "auto"
    ) -> Optional[Dict]:
        """
        对图片进行语义分割

        Args:
            image_path: 图片文件路径
            mask_prompt: 分割提示 ("auto" for automatic, or point coordinates)

        Returns:
            分割结果字典，包含:
            - masks: 掩码列表
            - scores: 置信度分数列表
            - boxes: 边界框列表
            - centers: 中心点列表
            - error: 错误信息 (如果失败)
        """
        if not self.client:
            return {
                "error": "Replicate client not initialized. Please check API token.",
                "masks": [],
                "scores": []
            }

        try:
            # 打开并验证图片
            if not os.path.exists(image_path):
                return {
                    "error": f"Image file not found: {image_path}",
                    "masks": [],
                    "scores": []
                }

            # 调用Replicate API
            with open(image_path, "rb") as f:
                output = self.client.run(
                    self.config.sam2_model_version,
                    input={
                        "image": f,
                        "mask_prompt": mask_prompt
                    }
                )

            # 解析输出（Replicate返回的格式取决于模型）
            # 这里需要根据实际的SAM 2 API响应格式调整
            return self._parse_replicate_output(output)

        except Exception as e:
            return {
                "error": f"Segmentation failed: {str(e)}",
                "masks": [],
                "scores": []
            }

    def _parse_replicate_output(self, output: any) -> Dict:
        """
        解析Replicate API的输出

        Args:
            output: Replicate API返回的原始输出

        Returns:
            标准化的分割结果字典
        """
        # 注意：这里的解析逻辑需要根据实际的Replicate SAM 2 API响应格式调整
        # SAM 2通常返回: masks, scores, logits (或类似的字段)

        result = {
            "masks": [],
            "scores": [],
            "boxes": [],
            "centers": []
        }

        # 如果输出是字典格式
        if isinstance(output, dict):
            # 提取masks和scores
            if 'masks' in output:
                result['masks'] = output['masks']
            if 'scores' in output:
                result['scores'] = output['scores']

        # 如果输出是列表格式
        elif isinstance(output, list):
            result['masks'] = output

        # 计算边界框和中心点
        if result['masks']:
            result['boxes'], result['centers'] = self._extract_boxes_and_centers(
                result['masks']
            )

        return result

    def _extract_boxes_and_centers(
        self,
        masks: List
    ) -> Tuple[List, List]:
        """
        从掩码提取边界框和中心点

        Args:
            masks: 掩码列表

        Returns:
            (boxes, centers) 元组
        """
        boxes = []
        centers = []

        for mask in masks:
            # 转换为numpy数组（如果还不是）
            if isinstance(mask, list):
                mask_array = np.array(mask)
            else:
                mask_array = mask

            # 确保是二维数组
            if mask_array.ndim == 1:
                # 如果是一维数组，跳过或返回默认值
                boxes.append((0, 0, 0, 0))
                centers.append((0, 0))
                continue

            # 计算边界框
            rows = np.any(mask_array, axis=1)
            cols = np.any(mask_array, axis=0)

            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                boxes.append((int(cmin), int(rmin), int(cmax), int(rmax)))

                # 计算中心点
                center_y = int(np.mean([rmin, rmax]))
                center_x = int(np.mean([cmin, cmax]))
                centers.append((center_x, center_y))
            else:
                boxes.append((0, 0, 0, 0))
                centers.append((0, 0))

        return boxes, centers

    def segment_and_crop(
        self,
        image_path: str,
        padding: int = 10
    ) -> Optional[Dict]:
        """
        分割图片并裁剪各个区域

        Args:
            image_path: 图片路径
            padding: 边界框扩展像素数

        Returns:
            包含裁剪图片的结果字典
        """
        # 先进行分割
        segmentation_result = self.segment_image(image_path)

        if "error" in segmentation_result:
            return segmentation_result

        # 加载原始图片
        image = Image.open(image_path)

        # 裁剪各个区域
        cropped_images = []
        boxes = segmentation_result.get('boxes', [])

        for box in boxes:
            x1, y1, x2, y2 = box

            # 添加padding并确保在图片范围内
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)

            # 裁剪
            cropped = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped)

        # 添加到结果中
        segmentation_result['cropped_images'] = cropped_images

        return segmentation_result

    def prepare_for_vlm(
        self,
        segmentation_result: Dict
    ) -> List[Dict]:
        """
        准备分割结果给VLM识别

        Args:
            segmentation_result: segment_image()返回的结果

        Returns:
            VLM输入列表，每个元素包含:
            - image: PIL.Image
            - center: (x, y) 中心点坐标
            - confidence: float 置信度
            - box: (x1, y1, x2, y2) 边界框
        """
        results = []

        cropped_images = segmentation_result.get('cropped_images', [])
        centers = segmentation_result.get('centers', [])
        scores = segmentation_result.get('scores', [])
        boxes = segmentation_result.get('boxes', [])

        for i, img in enumerate(cropped_images):
            center = centers[i] if i < len(centers) else (0, 0)
            score = scores[i] if i < len(scores) else 0.0
            box = boxes[i] if i < len(boxes) else (0, 0, 0, 0)

            results.append({
                'image': img,
                'center': center,
                'confidence': score,
                'box': box,
                'index': i
            })

        return results

    def export_masks(
        self,
        segmentation_result: Dict,
        output_dir: str
    ) -> List[str]:
        """
        导出掩码图片到文件

        Args:
            segmentation_result: 分割结果
            output_dir: 输出目录

        Returns:
            保存的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        masks = segmentation_result.get('masks', [])

        for i, mask in enumerate(masks):
            # 转换为图片
            if isinstance(mask, list):
                mask_array = np.array(mask, dtype=np.uint8) * 255
            else:
                mask_array = (mask * 255).astype(np.uint8)

            mask_image = Image.fromarray(mask_array, mode='L')

            # 保存
            path = os.path.join(output_dir, f"mask_{i}.png")
            mask_image.save(path)
            saved_paths.append(path)

        return saved_paths
