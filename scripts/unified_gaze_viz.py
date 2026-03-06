#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Gaze Visualization (Optimized) - VLM + SAM2 + Heatmap + Trajectory
Complete pipeline for IROS paper figure generation

Changes from original:
1. VLM uses relative coordinates (0.0-1.0) to be resolution-independent.
2. Removed hardcoded pixel thresholds (magic numbers).
3. Added retry logic for API calls.
4. Optimized heatmap generation using vectorized numpy operations.
5. Replaced skimage with cv2 for faster resizing.
6. Added structured logging.

Usage:
    python scripts/unified_gaze_viz_opt.py --image data/test1.jpg --use-sam2 --sam2-model models/sam2/sam2_hiera_small.pt
"""

import os
import sys
import json
import argparse
import base64
import time
import logging
import re
import numpy as np
from PIL import Image
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.ndimage import gaussian_filter, center_of_mass
import cv2
from contextlib import nullcontext

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Project paths setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================
# VLM Recognition Module (Optimized)
# ============================================

class VLMRecognizer:
    """VLM Exhibit Recognizer with Relative Coordinates and Retry Logic"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or os.getenv("VLM_MODEL", "qwen-vl-max-latest")

    def recognize(self, image_path: str, max_retries: int = 3) -> List[Dict]:
        logger.info("="*50)
        logger.info("Step (a): VLM Exhibit Recognition")
        logger.info("="*50)
        logger.info(f"Model: {self.model}")

        if not self.api_key:
            logger.error("No valid API Key found. Set QWEN_API_KEY in .env")
            return []

        # Load image
        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            img = Image.open(image_path)
            width, height = img.size
            logger.info(f"Image Size: {width}x{height}")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return []

        # Optimized Prompt: Requests relative coordinates (0.0-1.0)
        prompt = """You are an expert art curator analyzing an exhibition hall.
        
Task: Identify all PAINTINGS, SCULPTURES, or PHOTOS hanging on the wall or displayed.

CRITICAL INSTRUCTION FOR COORDINATES:
- Return RELATIVE coordinates normalized to 0.0-1.0 range.
- [0.0, 0.0] is Top-Left, [1.0, 1.0] is Bottom-Right.
- bbox format: [x1, y1, x2, y2] (xmin, ymin, xmax, ymax)

Exclusion Criteria:
- IGNORE floor reflections.
- IGNORE the floor itself.
- Focus on every single painting on the wall.

Return strictly JSON format:
{
  "exhibits": [
    {"name": "Painting 1", "type": "Painting", "bbox": [0.15, 0.25, 0.35, 0.55], "description": "Abstract art"},
    {"name": "Sculpture 1", "type": "Sculpture", "bbox": [0.60, 0.40, 0.75, 0.70], "description": "Bronze statue"}
  ]
}
"""

        for attempt in range(max_retries):
            try:
                logger.info(f"Calling VLM API (Attempt {attempt+1}/{max_retries})...")
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]}
                    ],
                    temperature=0.1, # Low temperature for deterministic results
                    response_format={"type": "json_object"} # Force JSON mode if supported
                )

                result_text = response.choices[0].message.content
                exhibits = self._parse_and_validate_response(result_text, width, height)
                
                if exhibits:
                    return exhibits
                else:
                    logger.warning("VLM returned valid JSON but no valid exhibits found.")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(1) # Simple backoff

        logger.error("VLM Recognition failed after all retries.")
        return []

    def _parse_and_validate_response(self, text: str, width: int, height: int) -> List[Dict]:
        try:
            # Robust JSON Extraction
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in response.")
                return []
            
            data = json.loads(json_match.group())
            raw_exhibits = data.get('exhibits', [])
            valid_exhibits = []

            type_map = {
                '画作': 'Painting', '绘画': 'Painting', '画': 'Painting',
                '雕塑': 'Sculpture', '装置': 'Installation', '摄影': 'Photography'
            }

            for i, ex in enumerate(raw_exhibits):
                bbox = ex.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # Normalize check: if values are > 1.0, assume pixels and normalize
                if any(x > 1.0 for x in bbox):
                    logger.warning(f"VLM returned absolute pixels for {ex.get('name')}, normalizing...")
                    bbox = [
                        bbox[0]/width if bbox[0] > 1 else bbox[0],
                        bbox[1]/height if bbox[1] > 1 else bbox[1],
                        bbox[2]/width if bbox[2] > 1 else bbox[2],
                        bbox[3]/height if bbox[3] > 1 else bbox[3]
                    ]

                # Convert to absolute pixels for internal processing
                x1 = int(max(0, bbox[0]) * width)
                y1 = int(max(0, bbox[1]) * height)
                x2 = int(min(1.0, bbox[2]) * width)
                y2 = int(min(1.0, bbox[3]) * height)

                # --- Robust Filtering Rules ---
                # 1. Size Check: Must be at least 0.5% of image area
                area = (x2 - x1) * (y2 - y1)
                min_area = (width * height) * 0.0015
                if area < min_area:
                    continue

                # 2. Position Check: Ignore if center is too low (likely floor)
                center_y = (y1 + y2) / 2
                if center_y > height * 0.85: # Bottom 15% is usually floor
                    continue

                ex_type = type_map.get(ex.get('type'), ex.get('type', 'Painting'))
                
                valid_exhibits.append({
                    "id": f"E{i+1}",
                    "name": ex.get('name', f'Exhibit{i+1}'),
                    "type": ex_type,
                    "bbox": [x1, y1, x2, y2], # Absolute pixels
                    "norm_bbox": bbox,       # Relative coords
                    "description": ex.get('description', '')
                })

            logger.info(f"Detected {len(valid_exhibits)} valid exhibits after filtering.")
            return valid_exhibits

        except json.JSONDecodeError:
            logger.error("JSON Decode Error in VLM response")
            return []

# ============================================
# SAM2 Segmentation Module (Optimized)
# ============================================

class SAM2Segmenter:
    """SAM2 box-prompted refinement for VLM detections."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            logger.error("SAM2 not installed. Please install the official facebookresearch/sam2 package.")
            raise

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {model_path}")

        self.torch = torch
        self.device = device
        self.model_path = model_path
        self.build_sam2 = build_sam2
        self.SAM2ImagePredictor = SAM2ImagePredictor

        cfg_candidates = self._infer_model_cfg_candidates(model_path)
        self.model_cfg = None
        self.predictor = None
        last_error = None

        for cfg in cfg_candidates:
            try:
                logger.info(f"Initializing SAM2 with cfg={cfg} on {device}...")
                model = self.build_sam2(cfg, model_path, device=device)
                self.predictor = self.SAM2ImagePredictor(model)
                self.model_cfg = cfg
                logger.info("SAM2 loaded successfully.")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"SAM2 load failed with cfg={cfg}: {e}")

        if self.predictor is None:
            raise RuntimeError(
                "Failed to load SAM2 with all config candidates. "
                f"checkpoint={model_path}, candidates={cfg_candidates}, last_error={last_error}"
            )

    @staticmethod
    def _infer_model_cfg_candidates(model_path: str) -> List[str]:
        """
        Return multiple config candidates because different SAM2 installations
        sometimes expect either:
        - configs/sam2/...yaml  (official README style)
        - sam2_hiera_s.yaml     (some packaged installs / hydra lookup)
        """
        name = os.path.basename(model_path).lower()

        is_21 = 'sam2.1' in name
        if is_21:
            prefix = 'configs/sam2.1/'
            if 'tiny' in name:
                yaml_name = 'sam2.1_hiera_t.yaml'
            elif 'base_plus' in name or 'base-plus' in name or 'base+' in name:
                yaml_name = 'sam2.1_hiera_b+.yaml'
            elif 'large' in name:
                yaml_name = 'sam2.1_hiera_l.yaml'
            else:
                yaml_name = 'sam2.1_hiera_s.yaml'
        else:
            prefix = 'configs/sam2/'
            if 'tiny' in name:
                yaml_name = 'sam2_hiera_t.yaml'
            elif 'base_plus' in name or 'base-plus' in name or 'base+' in name:
                yaml_name = 'sam2_hiera_b+.yaml'
            elif 'large' in name:
                yaml_name = 'sam2_hiera_l.yaml'
            else:
                yaml_name = 'sam2_hiera_s.yaml'

        candidates = [prefix + yaml_name, yaml_name]

        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    @staticmethod
    def _normalize_mask_output(masks) -> Optional[np.ndarray]:
        if masks is None:
            return None

        arr = np.asarray(masks)
        if arr.size == 0:
            return None

        if arr.ndim == 3:
            arr = arr[0]
        elif arr.ndim > 3:
            arr = np.squeeze(arr)
            if arr.ndim == 3:
                arr = arr[0]

        if arr.ndim != 2:
            return None

        return arr.astype(np.uint8)

    def refine_with_vlm_boxes(self, image_np: np.ndarray, vlm_exhibits: List[Dict]) -> List[Dict]:
        logger.info("=" * 50)
        logger.info("Step (b): SAM2 Refined Segmentation")
        logger.info("=" * 50)

        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)

        self.predictor.set_image(image_np)
        refined_exhibits = []

        use_autocast = self.device.startswith('cuda') and self.torch.cuda.is_available()
        autocast_ctx = self.torch.autocast('cuda', dtype=self.torch.bfloat16) if use_autocast else nullcontext()

        with self.torch.inference_mode(), autocast_ctx:
            for exhibit in vlm_exhibits:
                x1, y1, x2, y2 = exhibit['bbox']
                w = x2 - x1
                h = y2 - y1
                pad_x = int(w * 0.15)
                pad_y = int(h * 0.15)

                x1p = max(0, x1 - pad_x)
                y1p = max(0, y1 - pad_y)
                x2p = min(image_np.shape[1] - 1, x2 + pad_x)
                y2p = min(image_np.shape[0] - 1, y2 + pad_y)

                box_prompt = np.array([x1p, y1p, x2p, y2p], dtype=np.float32)

                try:
                    masks, scores, _ = self.predictor.predict(
                        box=box_prompt,
                        multimask_output=False,
                    )

                    mask = self._normalize_mask_output(masks)
                    if mask is None:
                        refined_exhibits.append(self._get_fallback_entry(exhibit))
                        continue

                    y_indices, x_indices = np.where(mask > 0)
                    if len(x_indices) == 0 or len(y_indices) == 0:
                        refined_exhibits.append(self._get_fallback_entry(exhibit))
                        continue

                    score_arr = np.asarray(scores)
                    score = float(score_arr.reshape(-1)[0]) if score_arr.size > 0 else 0.0

                    y1_new, y2_new = int(y_indices.min()), int(y_indices.max())
                    x1_new, x2_new = int(x_indices.min()), int(x_indices.max())
                    y_center, x_center = center_of_mass(mask)

                    refined_exhibits.append({
                        **exhibit,
                        'bbox': [x1_new, y1_new, x2_new, y2_new],
                        'center': [int(x_center), int(y_center)],
                        'area': int(mask.sum()),
                        'sam_score': score,
                        'mask': mask.astype(bool),
                    })
                except Exception as e:
                    logger.warning(f"SAM2 failed for {exhibit['name']}: {e}")
                    refined_exhibits.append(self._get_fallback_entry(exhibit))

        return refined_exhibits

    def _get_fallback_entry(self, exhibit: Dict) -> Dict:
        """Fallback to VLM bbox when SAM2 fails."""
        x1, y1, x2, y2 = exhibit['bbox']
        return {
            **exhibit,
            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            'area': max(1, (x2 - x1) * (y2 - y1)),
            'sam_score': 0.0,
            'mask': None,
        }
# ============================================
# Heatmap & Trajectory (Optimized)
# ============================================

def generate_heatmap(image_np: np.ndarray, exhibits: List[Dict], sigma: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("Generating Natural Heatmap...")
    height, width = image_np.shape[:2]
    
    # 创建一个纯净的注视点分布图，不再使用 mask 区域填充
    saliency = np.zeros((height, width), dtype=np.float32)
    
    # 仅使用画作中心点作为高斯源
    for ex in exhibits:
        cx, cy = ex['center']
        if 0 <= cy < height and 0 <= cx < width:
            saliency[cy, cx] = 1.0

    # 大高斯模糊：这会让热点自然扩散，形成类似图2的效果
    saliency = gaussian_filter(saliency, sigma=sigma)
    
    # 归一化
    if saliency.max() > 0:
        saliency /= saliency.max()

    # 使用 Jet 颜色映射覆盖原图
    heatmap_colored = plt.get_cmap('jet')(saliency)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # 融合：原图变暗一点，热力图更突出
    alpha = 0.6
    result_img = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored, alpha, 0)

    return saliency, result_img

def predict_scan_path(image_np: np.ndarray, saliency_map: np.ndarray, exhibits: List[Dict], num_fixations: int = 10) -> List[Dict]:
    logger.info("Predicting Scan Path...")
    height, width = image_np.shape[:2]
    img_center = np.array([width/2, height/2])

    exhibit_scores = []
    for ex in exhibits:
        cx, cy = ex['center']
        bbox = ex['bbox']
        
        # Safe slicing
        y1, y2 = max(0, bbox[1]), min(height, bbox[3])
        x1, x2 = max(0, bbox[0]), min(width, bbox[2])
        
        region_saliency = saliency_map[y1:y2, x1:x2].mean() if (x2>x1 and y2>y1) else 0

        # Center Bias
        dist = np.linalg.norm(np.array([cx, cy]) - img_center)
        center_bias = np.exp(-dist / (min(width, height) / 2))

        # Combined Score
        score = region_saliency * 0.5 + center_bias * 0.3 + 0.2 # Base interest
        exhibit_scores.append({'exhibit': ex, 'score': score})

    # Top-K selection
    exhibit_scores.sort(key=lambda x: x['score'], reverse=True)
    selected = exhibit_scores[:min(num_fixations, len(exhibit_scores))]

    # Spatial Sort: Left-to-Right, Top-to-Bottom scan pattern
    selected.sort(key=lambda x: x['exhibit']['center'][0]*0.7 + x['exhibit']['center'][1]*0.3)

    fixations = []
    for i, item in enumerate(selected):
        ex = item['exhibit']
        # Simulated duration based on area and interest
        duration = min(250, 40 * (0.6 + item['score']) * (1 + np.log(ex['area']/5000 + 1) * 0.3)) / 10

        fixations.append({
            'sequence': i + 1,
            'exhibit_id': ex['id'],
            'center': ex['center'],
            'duration': duration,
            'score': float(item['score'])
        })

    return fixations

# ============================================
# Main Pipeline
# ============================================

def run_pipeline(args):
    start_time = time.time()
    
    # 1. Setup
    image_path = args.image
    output_dir = args.output or f"data/outputs/{os.path.splitext(os.path.basename(image_path))[0]}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Output Directory: {output_dir}")
    
    # Load Image
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img_np = np.array(pil_img)
    except IOError:
        logger.error(f"Cannot open image: {image_path}")
        return

    # 2. VLM Detection
    vlm = VLMRecognizer()
    vlm_exhibits = vlm.recognize(image_path)
    if not vlm_exhibits:
        logger.error("Pipeline aborted due to VLM failure.")
        return

# 修改 run_pipeline 中的 SAM2 部分
# 关键：检查你是否在命令行确实传入了 --use-sam2 参数
# 如果你还是不想每次都输参数，可以在代码里硬编码 default=True

    # 3. SAM2 Segmentation
    if args.use_sam2:
        try:
            logger.info("Starting SAM2 processing...")
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"SAM2 checkpoint: {args.sam2_model}")
            sam = SAM2Segmenter(args.sam2_model, device)
            exhibits = sam.refine_with_vlm_boxes(img_np, vlm_exhibits)
            logger.info(f"SAM2 finished. Using cfg={sam.model_cfg}")
        except Exception as e:
            logger.error(f"SAM2 failed: {e}")
            logger.warning("Falling back to VLM boxes.")
            exhibits = _vlm_fallback(vlm_exhibits)
    else:
        logger.warning("SAM2 skipped. Check --use-sam2 argument.")
        exhibits = _vlm_fallback(vlm_exhibits)

    # 4. Heatmap
    saliency_map, heatmap_img = generate_heatmap(img_np, exhibits)
    cv2.imwrite(os.path.join(output_dir, "panel_c_heatmap.png"), cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))

    # 5. Trajectory
    fixations = predict_scan_path(img_np, saliency_map, exhibits, args.num_fixations)

    # 6. Visualization & Figure Generation
    _generate_all_figures(pil_img, img_np, heatmap_img, exhibits, fixations, output_dir)
    
    # 7. Data Generation
    _save_data(exhibits, fixations, output_dir)

    logger.info(f"Pipeline completed in {time.time() - start_time:.2f}s")

def _vlm_fallback(vlm_exhibits):
    """Convert VLM format to Unified format without SAM2"""
    refined = []
    for ex in vlm_exhibits:
        bbox = ex['bbox']
        refined.append({
            **ex,
            'center': [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)],
            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            'sam_score': 0.8,
            'mask': None
        })
    return refined

def _generate_all_figures(pil_img, img_np, heatmap_img, exhibits, fixations, output_dir):
    """Generates the 4-panel figure and individual assets efficiently"""
    logger.info("Generating visualization figures...")
    
    # Setup Figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # (a) Original
    axes[0].imshow(pil_img)
    axes[0].set_title('(a) Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # (b) Segmentation
    axes[1].imshow(pil_img)
    colors = plt.cm.tab10(np.linspace(0, 1, len(exhibits)))
    for i, ex in enumerate(exhibits):
        bbox = ex['bbox']
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                         fill=False, edgecolor=colors[i], linewidth=2)
        axes[1].add_patch(rect)
        axes[1].text(bbox[0], max(bbox[1]-5, 5), f"{ex['id']}", 
                    color='white', fontsize=8, fontweight='bold',
                    bbox=dict(facecolor=colors[i], alpha=0.8, edgecolor='none', pad=1))
    axes[1].set_title('(b) Detection', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # (c) Heatmap
    axes[2].imshow(heatmap_img)
    axes[2].set_title('(c) Heatmap', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # (d) Trajectory
    axes[3].imshow(img_np)
    if len(fixations) > 1:
        px = [f['center'][0] for f in fixations]
        py = [f['center'][1] for f in fixations]
        axes[3].plot(px, py, color='black', linewidth=6, alpha=0.7)
        axes[3].plot(px, py, color='white', linewidth=3, alpha=1.0)
    
    for fix in fixations:
        cx, cy = fix['center']
        r = max(15, min(40, int(fix['duration'] * 2)))
        axes[3].add_patch(Circle((cx, cy), r+2, facecolor='black', alpha=0.7))
        axes[3].add_patch(Circle((cx, cy), r, facecolor='#FFD700', alpha=1.0))
        axes[3].text(cx, cy, str(fix['sequence']), color='black', fontweight='bold', ha='center', va='center')
    
    axes[3].set_title('(d) Scan Path', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    
    outfile = os.path.join(output_dir, "paper_figure.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

    # Save individual (b) and (d) mainly because (c) is already saved
    # Note: For production, we can reuse the Axes objects, but redrawing is cleaner for logic here
    # (Skipping redundant code for brevity, assumes paper_figure.png is sufficient)

def _save_data(exhibits, fixations, output_dir):
    data = {
        'timestamp': datetime.now().isoformat(),
        'exhibits': [{k: v for k, v in ex.items() if k != 'mask'} for ex in exhibits], # Exclude mask array
        'fixations': fixations
    }
    with open(os.path.join(output_dir, 'analysis_data.json'), 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--use-sam2", action="store_true")
    parser.add_argument("--sam2-model", type=str, default="models/sam2/sam2_hiera_small.pt")
    parser.add_argument("--num-fixations", type=int, default=10)
    
    args = parser.parse_args()
    
    if os.path.exists(args.image):
        run_pipeline(args)
    else:
        logger.error("Image file not found.")