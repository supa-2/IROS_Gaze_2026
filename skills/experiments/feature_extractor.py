#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Extractor - 特征提取器

使用 VLM + SAM2 提取展品的结构化特征信息
"""

import os
import sys
import json
import base64
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class FeatureExtractor:
    """
    展品特征提取器

    使用 VLM + SAM2 提取展品的语义和空间特征
    """

    def __init__(self, vlm_model: str = "qwen-vl-max-latest", use_sam2: bool = False):
        """
        初始化特征提取器

        Args:
            vlm_model: VLM 模型名称
            use_sam2: 是否使用 SAM2 进行语义分割
        """
        self.vlm_model = vlm_model
        self.use_sam2 = use_sam2

        # 初始化 VLM 客户端
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )

        # 初始化 SAM2（如果需要）
        self.sam2_segmenter = None
        if use_sam2:
            try:
                from skills.segmentation.sam2_local import SAM2LocalSegmenter
                self.sam2_segmenter = SAM2LocalSegmenter(model_size='small', device='cpu')
                print("[+] SAM2 initialized for feature extraction")
            except Exception as e:
                print(f"[!] SAM2 initialization failed: {e}")
                self.use_sam2 = False

    def extract_exhibit_features(
        self,
        image_path: str,
        exhibit_id: str,
        exhibit_name: str = None,
        bbox: List[int] = None
    ) -> Dict:
        """
        提取单个展品的特征

        Args:
            image_path: 图片路径
            exhibit_id: 展品ID
            exhibit_name: 展品名称（可选）
            bbox: 边界框 [x1, y1, x2, y2]（可选）

        Returns:
            特征字典
        """
        features = {
            'exhibit_id': exhibit_id,
            'exhibit_name': exhibit_name or exhibit_id,
            'bbox': bbox,
            'semantic_features': {},
            'spatial_features': {},
            'visual_features': {}
        }

        # 1. SAM2 分割（如果启用）
        if self.use_sam2 and self.sam2_segmenter:
            segmentation_result = self._segment_exhibit(image_path, bbox)
            features['visual_features'] = {
                'segmentation_area': segmentation_result.get('area', 0),
                'segmentation_score': segmentation_result.get('score', 0.0),
                'center_point': segmentation_result.get('center', (0, 0))
            }

        # 2. VLM 语义分析
        semantic_features = self._analyze_semantics(image_path, bbox, exhibit_name)
        features['semantic_features'] = semantic_features

        return features

    def extract_surrounding_features(
        self,
        image_path: str,
        current_exhibit_id: str,
        neighbor_exhibits: List[Dict],
        topology_info: Dict = None
    ) -> Dict:
        """
        提取周边展品的综合特征（用于预测）

        Args:
            image_path: 图片路径
            current_exhibit_id: 当前展品ID
            neighbor_exhibits: 邻居展品列表 [{'id':, 'name':, 'relation':, 'bbox':}]
            topology_info: 拓扑信息

        Returns:
            结构化特征字典，用于喂给预测模型
        """
        context = {
            'current_exhibit': current_exhibit_id,
            'image_path': image_path,
            'neighbors': [],
            'topology_summary': topology_info or {}
        }

        # 提取每个邻居的特征
        for neighbor in neighbor_exhibits:
            neighbor_feat = self.extract_exhibit_features(
                image_path=image_path,
                exhibit_id=neighbor.get('id'),
                exhibit_name=neighbor.get('name'),
                bbox=neighbor.get('bbox')
            )
            neighbor_feat['relation'] = neighbor.get('relation', 'unknown')
            context['neighbors'].append(neighbor_feat)

        # 生成整体场景描述
        context['scene_description'] = self._generate_scene_description(context)

        return context

    def _segment_exhibit(self, image_path: str, bbox: List[int] = None) -> Dict:
        """使用 SAM2 分割展品"""
        if not self.sam2_segmenter:
            return {}

        try:
            # 如果有 bbox，使用 point prompt
            # 否则使用自动分割
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)

            if bbox:
                # 使用 bbox 作为提示
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                masks, scores, _ = self.sam2_segmenter.predict(
                    point_coords=np.array([[center_x, center_y]]),
                    point_labels=np.array([1]),
                    box=np.array([x1, y1, x2, y2])
                )

                if len(masks) > 0:
                    mask = masks[0]
                    area = int(mask.sum())
                    return {
                        'area': area,
                        'score': float(scores[0]),
                        'center': (center_x, center_y)
                    }
        except Exception as e:
            print(f"[!] SAM2 segmentation failed: {e}")

        return {}

    def _analyze_semantics(
        self,
        image_path: str,
        bbox: List[int] = None,
        exhibit_name: str = None
    ) -> Dict:
        """使用 VLM 分析语义特征"""
        # 读取并编码图片
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        # 构建 prompt
        prompt = """分析这个展品，提取以下特征（以JSON格式返回）：

{
  "category": "展品类型（如：绘画/雕塑/装置/文物/说明牌）",
  "era": "年代（如：古代/现代/当代，如果无法判断则为unknown）",
  "theme": "主题（简短描述）",
  "visual_salience": "视觉显著性（high/medium/low）",
  "interaction_type": "交互类型（观看/阅读/互动）",
  "estimated_engagement": "预估参与度（A/B/C/D/E）"
}

只返回JSON，不要其他内容。"""

        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }]

            response = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )

            result_text = response.choices[0].message.content

            # 解析 JSON
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            return {'category': 'unknown', 'theme': 'unknown'}

        except Exception as e:
            print(f"[!] VLM analysis failed: {e}")
            return {'category': 'unknown', 'theme': 'unknown'}

    def _generate_scene_description(self, context: Dict) -> str:
        """生成场景描述"""
        current = context.get('current_exhibit', 'Unknown')
        neighbors = context.get('neighbors', [])

        description = f"当前在 {current}，"

        if neighbors:
            neighbor_types = [n.get('semantic_features', {}).get('category', 'exhibit')
                            for n in neighbors[:3]]
            description += f" 周边有 {', '.join(neighbor_types)} 等展品。"

        # 添加空间关系
        relations = [n.get('relation', '') for n in neighbors if n.get('relation')]
        if relations:
            if 'next' in relations:
                description += " 沿主路径继续前进。"
            if 'visual' in relations:
                description += " 有视觉连接的展品。"

        return description

    def save_features(self, features: Dict, output_path: str):
        """保存特征到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2, ensure_ascii=False)
        print(f"[+] Features saved to {output_path}")

    def load_features(self, feature_path: str) -> Dict:
        """从文件加载特征"""
        with open(feature_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# ============================================
# 批量特征提取
# ============================================

def batch_extract_features(
    image_path: str,
    exhibits: List[Dict],
    output_dir: str = "data/outputs/features"
) -> Dict[str, Dict]:
    """
    批量提取展品特征

    Args:
        image_path: 图片路径
        exhibits: 展品列表 [{'id':, 'name':, 'bbox':}]
        output_dir: 输出目录

    Returns:
        展品ID到特征的映射
    """
    extractor = FeatureExtractor(use_sam2=True)
    all_features = {}

    os.makedirs(output_dir, exist_ok=True)

    for exhibit in exhibits:
        exhibit_id = exhibit.get('id')
        print(f"[*] Extracting features for {exhibit_id}...")

        features = extractor.extract_exhibit_features(
            image_path=image_path,
            exhibit_id=exhibit_id,
            exhibit_name=exhibit.get('name'),
            bbox=exhibit.get('bbox')
        )

        all_features[exhibit_id] = features

        # 保存单个展品的特征
        feature_path = os.path.join(output_dir, f"{exhibit_id}_features.json")
        extractor.save_features(features, feature_path)

    # 保存汇总
    summary_path = os.path.join(output_dir, "all_features.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_features, f, indent=2, ensure_ascii=False)

    print(f"[+] Saved {len(all_features)} exhibit features to {output_dir}")

    return all_features


if __name__ == "__main__":
    # 测试特征提取
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/R.jpg")
    parser.add_argument("--exhibits", default="data/outputs/vlm/vlm_result.json")
    parser.add_argument("--output", default="data/outputs/features")
    parser.add_argument("--use-sam2", action="store_true")
    args = parser.parse_args()

    # 加载展品列表
    with open(args.exhibits, 'r') as f:
        data = json.load(f)
    exhibits = data.get('exhibits', [])

    # 批量提取
    batch_extract_features(args.image, exhibits, args.output)
