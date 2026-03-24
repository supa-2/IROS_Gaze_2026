#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Gaze Visualization (Enhanced) - VLM + SAM2 + Heatmap + Trajectory
Complete pipeline for IROS paper figure generation.

Enhancements:
1. VLM prompt focuses on wall-mounted paintings, framed photos, and similar 2D wall exhibits.
2. Parser filtering is tuned for wall-mounted exhibits only.
3. SAM2 box refinement keeps the original logic but adds prompt padding for tighter, more stable masks.
4. Heatmap now uses both exhibit centers and a light mask prior for more natural saliency.
5. Scan path changed from rigid spatial sorting to a transition-aware path planner.
6. Detection panel can display semi-transparent SAM2 masks.
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
from typing import List, Dict, Tuple
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
# VLM Recognition Module
# ============================================

class VLMRecognizer:
    """VLM exhibit recognizer with relative coordinates and retry logic."""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or os.getenv("VLM_MODEL", "qwen-vl-max-latest")

    def recognize(self, image_path: str, max_retries: int = 3) -> List[Dict]:
        logger.info("=" * 50)
        logger.info("Step (a): VLM Exhibit Recognition")
        logger.info("=" * 50)
        logger.info(f"Model: {self.model}")

        if not self.api_key:
            logger.error("No valid API Key found. Set QWEN_API_KEY in .env")
            return []

        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            img = Image.open(image_path)
            width, height = img.size
            logger.info(f"Image Size: {width}x{height}")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return []

        prompt = """You are an expert museum curator analyzing an exhibition hall.

Task: Detect EVERY visible wall-mounted exhibit in the scene, focusing on:
1. Paintings
2. Framed photos
3. Posters
4. Other flat artworks mounted on walls or partition panels

CRITICAL INSTRUCTION FOR COORDINATES:
- Return RELATIVE coordinates normalized to the 0.0-1.0 range.
- [0.0, 0.0] is Top-Left, [1.0, 1.0] is Bottom-Right.
- bbox format: [x1, y1, x2, y2] (xmin, ymin, xmax, ymax)

IMPORTANT RULES:
- Focus on every single painting or framed wall-mounted artwork that is visible.
- Include small side-wall artworks if visible.
- Do NOT detect sculptures, statues, floor-standing objects, display cases, or installations.
- Do NOT merge multiple artworks into one bounding box.
- Ignore floor reflections, glare, empty walls, and empty architectural surfaces.
- Prefer precise boxes around the actual artwork rather than the whole partition wall.

Return strictly JSON format:
{
  "exhibits": [
    {
      "name": "Painting 1",
      "type": "Painting",
      "bbox": [0.15, 0.25, 0.35, 0.55],
      "description": "Framed painting on wall"
    },
    {
      "name": "Photo 1",
      "type": "Photography",
      "bbox": [0.60, 0.30, 0.72, 0.58],
      "description": "Framed photo on side wall"
    }
  ]
}
"""

        for attempt in range(max_retries):
            try:
                logger.info(f"Calling VLM API (Attempt {attempt + 1}/{max_retries})...")
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            ],
                        }
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )

                result_text = response.choices[0].message.content
                exhibits = self._parse_and_validate_response(result_text, width, height)

                if exhibits:
                    return exhibits
                logger.warning("VLM returned valid JSON but no valid exhibits found.")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(1)

        logger.error("VLM Recognition failed after all retries.")
        return []

    def _parse_and_validate_response(self, text: str, width: int, height: int) -> List[Dict]:
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in response.")
                return []

            data = json.loads(json_match.group())
            raw_exhibits = data.get("exhibits", [])
            valid_exhibits = []

            type_map = {
                "画作": "Painting", "绘画": "Painting", "画": "Painting",
                "painting": "Painting", "artwork": "Painting", "poster": "Painting",
                "摄影": "Photography", "照片": "Photography",
                "photo": "Photography", "photograph": "Photography", "photography": "Photography",
            }

            for i, ex in enumerate(raw_exhibits):
                bbox = ex.get("bbox", [])
                if len(bbox) != 4:
                    continue

                if any(x > 1.0 for x in bbox):
                    logger.warning(f"VLM returned absolute pixels for {ex.get('name')}, normalizing...")
                    bbox = [
                        bbox[0] / width if bbox[0] > 1 else bbox[0],
                        bbox[1] / height if bbox[1] > 1 else bbox[1],
                        bbox[2] / width if bbox[2] > 1 else bbox[2],
                        bbox[3] / height if bbox[3] > 1 else bbox[3],
                    ]

                bbox = [float(v) for v in bbox]
                x1 = int(max(0.0, min(1.0, bbox[0])) * width)
                y1 = int(max(0.0, min(1.0, bbox[1])) * height)
                x2 = int(max(0.0, min(1.0, bbox[2])) * width)
                y2 = int(max(0.0, min(1.0, bbox[3])) * height)

                if x2 <= x1 or y2 <= y1:
                    continue

                raw_type = str(ex.get("type", "")).strip()
                raw_type_lower = raw_type.lower()
                ex_type = type_map.get(raw_type_lower, type_map.get(raw_type, "Painting"))

                # Only keep wall-mounted 2D exhibits
                if ex_type not in {"Painting", "Photography"}:
                    continue

                area = max(1, (x2 - x1) * (y2 - y1))
                center_y = (y1 + y2) / 2

                min_area = (width * height) * 0.0015
                if area < min_area:
                    continue

                # Filter out objects that are too low in the image for wall art
                if center_y > height * 0.88:
                    continue

                valid_exhibits.append(
                    {
                        "id": f"E{i + 1}",
                        "name": ex.get("name", f"Exhibit{i + 1}"),
                        "type": ex_type,
                        "bbox": [x1, y1, x2, y2],
                        "norm_bbox": bbox,
                        "description": ex.get("description", ""),
                    }
                )

            logger.info(f"Detected {len(valid_exhibits)} valid exhibits after filtering.")
            return valid_exhibits

        except json.JSONDecodeError:
            logger.error("JSON Decode Error in VLM response")
            return []


# ============================================
# SAM2 Segmentation Module
# ============================================

class SAM2Segmenter:
    """SAM2 box-prompted refinement for VLM detections."""

    def __init__(self, model_path: str, device: str = "cuda"):
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
                f"Last error: {last_error}"
            )

    def _infer_model_cfg_candidates(self, model_path: str) -> List[str]:
        model_name = os.path.basename(model_path).lower()
        primary = self._infer_primary_cfg(model_name)

        candidates = [primary]
        basename = os.path.basename(primary)
        if basename not in candidates:
            candidates.append(basename)
        if model_name.startswith("sam2.1"):
            alt = primary.replace("configs/sam2.1/", "")
            if alt not in candidates:
                candidates.append(alt)
        else:
            alt = primary.replace("configs/sam2/", "")
            if alt not in candidates:
                candidates.append(alt)
        return candidates

    def _infer_primary_cfg(self, name: str) -> str:
        if "sam2.1" in name:
            prefix = "configs/sam2.1/"
            if "tiny" in name:
                return prefix + "sam2.1_hiera_t.yaml"
            if "base_plus" in name or "base-plus" in name:
                return prefix + "sam2.1_hiera_b+.yaml"
            if "large" in name:
                return prefix + "sam2.1_hiera_l.yaml"
            return prefix + "sam2.1_hiera_s.yaml"

        prefix = "configs/sam2/"
        if "tiny" in name:
            return prefix + "sam2_hiera_t.yaml"
        if "base_plus" in name or "base-plus" in name:
            return prefix + "sam2_hiera_b+.yaml"
        if "large" in name:
            return prefix + "sam2_hiera_l.yaml"
        return prefix + "sam2_hiera_s.yaml"

    @staticmethod
    def _normalize_mask_output(masks) -> np.ndarray:
        arr = np.asarray(masks)
        if arr.size == 0:
            return None
        if arr.ndim == 4:
            arr = arr[0, 0]
        elif arr.ndim == 3:
            arr = arr[0]
        elif arr.ndim == 2:
            pass
        else:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None
        return (arr > 0).astype(np.uint8)

    def refine_with_vlm_boxes(self, image_np: np.ndarray, vlm_exhibits: List[Dict]) -> List[Dict]:
        logger.info("=" * 50)
        logger.info("Step (b): SAM2 Refined Segmentation")
        logger.info("=" * 50)

        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)

        self.predictor.set_image(image_np)
        refined_exhibits = []

        use_autocast = self.device.startswith("cuda") and self.torch.cuda.is_available()
        autocast_ctx = self.torch.autocast("cuda", dtype=self.torch.bfloat16) if use_autocast else nullcontext()

        with self.torch.inference_mode(), autocast_ctx:
            for exhibit in vlm_exhibits:
                x1, y1, x2, y2 = exhibit["bbox"]
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
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

                    refined_exhibits.append(
                        {
                            **exhibit,
                            "bbox": [x1_new, y1_new, x2_new, y2_new],
                            "center": [int(x_center), int(y_center)],
                            "area": int(mask.sum()),
                            "sam_score": score,
                            "mask": mask.astype(bool),
                        }
                    )
                except Exception as e:
                    logger.warning(f"SAM2 failed for {exhibit['name']}: {e}")
                    refined_exhibits.append(self._get_fallback_entry(exhibit))

        return refined_exhibits

    def _get_fallback_entry(self, exhibit: Dict) -> Dict:
        x1, y1, x2, y2 = exhibit["bbox"]
        return {
            **exhibit,
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "area": max(1, (x2 - x1) * (y2 - y1)),
            "sam_score": 0.0,
            "mask": None,
        }


# ============================================
# Heatmap & Trajectory
# ============================================

def generate_heatmap(image_np: np.ndarray, exhibits: List[Dict], sigma: int = 55) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("Generating Natural Heatmap...")
    height, width = image_np.shape[:2]
    saliency = np.zeros((height, width), dtype=np.float32)

    for ex in exhibits:
        cx, cy = ex["center"]
        ex_type = ex.get("type", "Painting")
        weight = {
            "Painting": 1.00,
            "Photography": 0.95,
            "Sculpture": 1.20,
            "Installation": 1.10,
        }.get(ex_type, 1.0)

        mask = ex.get("mask")
        if mask is not None:
            saliency += mask.astype(np.float32) * (0.08 * weight)

        if 0 <= cy < height and 0 <= cx < width:
            saliency[cy, cx] += 1.0 * weight

    saliency = gaussian_filter(saliency, sigma=sigma)

    if saliency.max() > 0:
        saliency /= saliency.max()

    heatmap_colored = plt.get_cmap("jet")(saliency)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    alpha = 0.6
    result_img = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored, alpha, 0)
    return saliency, result_img


def predict_scan_path(image_np: np.ndarray, saliency_map: np.ndarray, exhibits: List[Dict], num_fixations: int = 10) -> List[Dict]:
    logger.info("Predicting Scan Path...")
    if not exhibits:
        return []

    height, width = image_np.shape[:2]
    img_center = np.array([width / 2, height / 2], dtype=np.float32)

    candidates = []
    for ex in exhibits:
        cx, cy = ex["center"]
        bbox = ex["bbox"]

        y1, y2 = max(0, bbox[1]), min(height, bbox[3])
        x1, x2 = max(0, bbox[0]), min(width, bbox[2])
        region_saliency = saliency_map[y1:y2, x1:x2].mean() if (x2 > x1 and y2 > y1) else 0.0

        dist_center = np.linalg.norm(np.array([cx, cy], dtype=np.float32) - img_center)
        center_bias = np.exp(-dist_center / (min(width, height) / 2))

        type_bonus = {
            "Painting": 0.03,
            "Photography": 0.02,
            "Sculpture": 0.10,
            "Installation": 0.08,
        }.get(ex.get("type", "Painting"), 0.0)

        size_bonus = min(0.15, np.log1p(max(1, ex["area"]) / 4000.0) * 0.06)
        base_score = 0.50 * region_saliency + 0.25 * center_bias + 0.15 + type_bonus + size_bonus
        candidates.append({"exhibit": ex, "base_score": float(base_score)})

    k = min(num_fixations, len(candidates))
    remaining = candidates.copy()

    start = max(
        remaining,
        key=lambda it: it["base_score"] + 0.20 * np.exp(
            -np.linalg.norm(np.array(it["exhibit"]["center"], dtype=np.float32) - img_center) /
            (0.35 * min(width, height))
        )
    )
    path = [start]
    remaining.remove(start)

    prev = np.array(start["exhibit"]["center"], dtype=np.float32)
    prev_dir = None

    while remaining and len(path) < k:
        best_idx = None
        best_value = -1e9

        for idx, it in enumerate(remaining):
            cur = np.array(it["exhibit"]["center"], dtype=np.float32)
            delta = cur - prev
            step_dist = np.linalg.norm(delta) + 1e-6

            move_penalty = step_dist / (0.60 * max(width, height))

            smooth_bonus = 0.0
            if prev_dir is not None:
                smooth_bonus = 0.10 * max(0.0, np.dot(delta / step_dist, prev_dir))

            row_bonus = 0.05 * np.exp(-abs(cur[1] - prev[1]) / (0.18 * height))
            value = it["base_score"] - 0.28 * move_penalty + smooth_bonus + row_bonus

            if value > best_value:
                best_value = value
                best_idx = idx

        chosen = remaining.pop(best_idx)
        cur = np.array(chosen["exhibit"]["center"], dtype=np.float32)
        delta = cur - prev
        norm = np.linalg.norm(delta)
        if norm > 1e-6:
            prev_dir = delta / norm
        prev = cur

        chosen["path_score"] = float(best_value)
        path.append(chosen)

    fixations = []
    for i, item in enumerate(path):
        ex = item["exhibit"]
        duration = 1.2 + 2.8 * item["base_score"]
        if ex.get("type") in {"Sculpture", "Installation"}:
            duration *= 1.10
        duration += 0.25 * np.log1p(max(1, ex["area"]) / 5000.0)
        duration = float(np.clip(duration, 1.0, 6.5))

        fixations.append(
            {
                "sequence": i + 1,
                "exhibit_id": ex["id"],
                "center": ex["center"],
                "duration": duration,
                "score": float(item["base_score"]),
            }
        )

    return fixations


# ============================================
# Main Pipeline
# ============================================

def run_pipeline(args, image_path=None, output_dir=None):
    start_time = time.time()

    image_path = image_path or args.image
    if image_path is None:
        logger.error("No input image provided.")
        return False

    output_dir = output_dir or args.output or f"data/outputs/{os.path.splitext(os.path.basename(image_path))[0]}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Input Image: {image_path}")
    logger.info(f"Output Directory: {output_dir}")

    try:
        pil_img = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_img)
    except IOError:
        logger.error(f"Cannot open image: {image_path}")
        return False

    vlm = VLMRecognizer()
    vlm_exhibits = vlm.recognize(image_path)
    if not vlm_exhibits:
        logger.error("Pipeline aborted due to VLM failure.")
        return False

    if args.use_sam2:
        try:
            logger.info("Starting SAM2 processing...")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
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

    saliency_map, heatmap_img = generate_heatmap(img_np, exhibits)
    cv2.imwrite(os.path.join(output_dir, "panel_c_heatmap.png"), cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))

    fixations = predict_scan_path(img_np, saliency_map, exhibits, args.num_fixations)

    _generate_all_figures(pil_img, img_np, heatmap_img, exhibits, fixations, output_dir, gif_duration_ms=args.gif_duration_ms, gif_hold_last_ms=args.gif_hold_last_ms)
    _save_data(exhibits, fixations, output_dir)

    logger.info(f"Pipeline completed in {time.time() - start_time:.2f}s")
    return True


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _list_images(input_dir: str):
    files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in SUPPORTED_IMAGE_EXTS:
            files.append(path)
    return files


def run_batch(args):
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    image_paths = _list_images(input_dir)
    if not image_paths:
        logger.error(f"No supported images found in: {input_dir}")
        return

    output_root = args.output or os.path.join("data", "outputs", os.path.basename(os.path.normpath(input_dir)))
    os.makedirs(output_root, exist_ok=True)

    logger.info("=" * 50)
    logger.info(f"Batch Mode: found {len(image_paths)} images in {input_dir}")
    logger.info(f"Batch Output Root: {output_root}")
    logger.info("=" * 50)

    success = 0
    failed = 0

    for idx, image_path in enumerate(image_paths, start=1):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        item_output_dir = os.path.join(output_root, stem)

        logger.info("")
        logger.info("#" * 70)
        logger.info(f"[{idx}/{len(image_paths)}] Processing: {image_path}")
        logger.info("#" * 70)

        ok = run_pipeline(args, image_path=image_path, output_dir=item_output_dir)
        if ok:
            success += 1
        else:
            failed += 1

    summary = {
        "input_dir": input_dir,
        "output_root": output_root,
        "total_images": len(image_paths),
        "success": success,
        "failed": failed,
        "images": [os.path.basename(p) for p in image_paths],
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_root, "batch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 50)
    logger.info(f"Batch finished. Success: {success}, Failed: {failed}, Total: {len(image_paths)}")
    logger.info(f"Summary saved to: {os.path.join(output_root, 'batch_summary.json')}")
    logger.info("=" * 50)


def _vlm_fallback(vlm_exhibits: List[Dict]) -> List[Dict]:
    refined = []
    for ex in vlm_exhibits:
        bbox = ex["bbox"]
        refined.append(
            {
                **ex,
                "center": [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)],
                "area": max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                "sam_score": 0.8,
                "mask": None,
            }
        )
    return refined


def _get_fixation_ordered_exhibits(exhibits, fixations):
    ex_by_id = {ex["id"]: ex for ex in exhibits}
    ordered = []
    used = set()
    for fix in fixations:
        ex = ex_by_id.get(fix["exhibit_id"])
        if ex is not None and ex["id"] not in used:
            ordered.append(ex)
            used.add(ex["id"])
    return ordered



def _render_detection_panel(ax, base_img, exhibits_subset, title="(b) Detection"):
    ax.imshow(base_img)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(exhibits_subset))))
    for i, ex in enumerate(exhibits_subset):
        color = colors[i]
        mask = ex.get("mask")
        if mask is not None:
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[..., 0] = color[0]
            overlay[..., 1] = color[1]
            overlay[..., 2] = color[2]
            overlay[..., 3] = mask.astype(np.float32) * 0.28
            ax.imshow(overlay)

        bbox = ex["bbox"]
        rect = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)
        label = f"{ex['id']} ({ex.get('type', 'Exhibit')[0]})"
        ax.text(
            bbox[0],
            max(bbox[1] - 5, 5),
            label,
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=1),
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")



def _render_scanpath_panel(ax, base_img, fixations_subset, title="(d) Scan Path"):
    ax.imshow(base_img)
    if len(fixations_subset) > 1:
        px = [f["center"][0] for f in fixations_subset]
        py = [f["center"][1] for f in fixations_subset]
        ax.plot(px, py, color="black", linewidth=6, alpha=0.65)
        ax.plot(px, py, color="white", linewidth=3, alpha=1.0)

    for fix in fixations_subset:
        cx, cy = fix["center"]
        r = max(15, min(40, int(fix["duration"] * 5)))
        ax.add_patch(Circle((cx, cy), r + 2, facecolor="black", alpha=0.7))
        ax.add_patch(Circle((cx, cy), r, facecolor="#FFD700", alpha=1.0))
        ax.text(cx, cy, str(fix["sequence"]), color="black", fontweight="bold", ha="center", va="center")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")



def _build_cumulative_heatmap(base_img_np, ordered_exhibits, sigma=55):
    height, width = base_img_np.shape[:2]
    saliency = np.zeros((height, width), dtype=np.float32)

    for ex in ordered_exhibits:
        cx, cy = ex["center"]
        ex_type = ex.get("type", "Painting")
        weight = {
            "Painting": 1.00,
            "Photography": 0.95,
            "Sculpture": 1.20,
            "Installation": 1.10,
        }.get(ex_type, 1.0)

        mask = ex.get("mask")
        if mask is not None:
            saliency += mask.astype(np.float32) * (0.08 * weight)

        if 0 <= cy < height and 0 <= cx < width:
            saliency[cy, cx] += 1.0 * weight

    saliency = gaussian_filter(saliency, sigma=sigma)
    if saliency.max() > 0:
        saliency /= saliency.max()

    heatmap_colored = plt.get_cmap("jet")(saliency)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    result_img = cv2.addWeighted(base_img_np, 0.4, heatmap_colored, 0.6, 0)
    return saliency, result_img




def _save_gif_from_pngs(frame_dir: str, prefix: str, out_name: str, duration_ms: int = 450, hold_last_ms: int = 1200):
    frame_paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(".png") and f.startswith(prefix)
    ])
    if not frame_paths:
        return None

    frames = [Image.open(fp).convert("P", palette=Image.Palette.ADAPTIVE) for fp in frame_paths]
    durations = [duration_ms] * len(frames)
    durations[-1] = hold_last_ms

    out_path = os.path.join(frame_dir, out_name)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    for fr in frames:
        fr.close()
    return out_path


def _save_progressive_gifs(output_dir: str, duration_ms: int = 450, hold_last_ms: int = 1200):
    gif_dir = os.path.join(output_dir, "progressive_gifs")
    os.makedirs(gif_dir, exist_ok=True)

    groups = [
        (os.path.join(output_dir, "progressive_scanpath"), "scanpath_", "scanpath.gif"),
        (os.path.join(output_dir, "progressive_heatmap"), "heatmap_", "heatmap.gif"),
        (os.path.join(output_dir, "progressive_segmentation"), "segmentation_", "segmentation.gif"),
    ]

    saved = {}
    for frame_dir, prefix, out_name in groups:
        tmp_gif = _save_gif_from_pngs(frame_dir, prefix, out_name, duration_ms=duration_ms, hold_last_ms=hold_last_ms)
        if tmp_gif is not None:
            final_path = os.path.join(gif_dir, out_name)
            if os.path.abspath(tmp_gif) != os.path.abspath(final_path):
                from shutil import copyfile
                copyfile(tmp_gif, final_path)
                os.remove(tmp_gif)
            saved[out_name] = os.path.relpath(final_path, output_dir)
    return saved


def _save_progressive_outputs(pil_img, img_np, exhibits, fixations, output_dir, gif_duration_ms=450, gif_hold_last_ms=1200):
    logger.info("Generating progressive outputs...")
    if not fixations:
        return

    ordered_exhibits = _get_fixation_ordered_exhibits(exhibits, fixations)
    scan_dir = os.path.join(output_dir, "progressive_scanpath")
    heat_dir = os.path.join(output_dir, "progressive_heatmap")
    seg_dir = os.path.join(output_dir, "progressive_segmentation")
    os.makedirs(scan_dir, exist_ok=True)
    os.makedirs(heat_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    for step in range(1, len(fixations) + 1):
        fix_subset = fixations[:step]
        ex_subset = ordered_exhibits[:step]

        # progressive scan path
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _render_scanpath_panel(ax, img_np, fix_subset, title=f"Scan Path {step}/{len(fixations)}")
        plt.tight_layout()
        plt.savefig(os.path.join(scan_dir, f"scanpath_{step:02d}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # progressive heatmap
        _, heat_img_step = _build_cumulative_heatmap(img_np, ex_subset)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(heat_img_step)
        ax.set_title(f"Heatmap {step}/{len(fixations)}", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(heat_dir, f"heatmap_{step:02d}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # progressive segmentation
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _render_detection_panel(ax, pil_img, ex_subset, title=f"Segmentation {step}/{len(fixations)}")
        plt.tight_layout()
        plt.savefig(os.path.join(seg_dir, f"segmentation_{step:02d}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    gif_map = _save_progressive_gifs(output_dir, duration_ms=gif_duration_ms, hold_last_ms=gif_hold_last_ms)
    if gif_map:
        logger.info(f"Saved progressive GIFs: {gif_map}")



def _generate_all_figures(pil_img, img_np, heatmap_img, exhibits, fixations, output_dir, gif_duration_ms=450, gif_hold_last_ms=1200):
    logger.info("Generating visualization figures...")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(pil_img)
    axes[0].set_title("(a) Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    _render_detection_panel(axes[1], pil_img, exhibits, title="(b) Detection")

    axes[2].imshow(heatmap_img)
    axes[2].set_title("(c) Heatmap", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    _render_scanpath_panel(axes[3], img_np, fixations, title="(d) Scan Path")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)

    outfile = os.path.join(output_dir, "paper_figure.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    _save_progressive_outputs(pil_img, img_np, exhibits, fixations, output_dir, gif_duration_ms=gif_duration_ms, gif_hold_last_ms=gif_hold_last_ms)


def _save_data(exhibits, fixations, output_dir):
    data = {
        "timestamp": datetime.now().isoformat(),
        "exhibits": [{k: v for k, v in ex.items() if k != "mask"} for ex in exhibits],
        "fixations": fixations,
        "progressive_outputs": {
            "scanpath_dir": "progressive_scanpath",
            "heatmap_dir": "progressive_heatmap",
            "segmentation_dir": "progressive_segmentation",
            "gif_dir": "progressive_gifs",
        },
    }
    with open(os.path.join(output_dir, "analysis_data.json"), "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Single input image path")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory for batch processing")
    parser.add_argument("--output", type=str, default=None, help="Single-image output dir or batch output root")
    parser.add_argument("--use-sam2", action="store_true")
    parser.add_argument("--sam2-model", type=str, default="models/sam2/sam2_hiera_small.pt")
    parser.add_argument("--num-fixations", type=int, default=10)
    parser.add_argument("--gif-duration-ms", type=int, default=450, help="Per-frame duration for progressive GIFs")
    parser.add_argument("--gif-hold-last-ms", type=int, default=1200, help="Final-frame hold duration for progressive GIFs")

    args = parser.parse_args()

    if bool(args.image) == bool(args.input_dir):
        logger.error("Please provide exactly one of --image or --input-dir.")
    elif args.image:
        if os.path.exists(args.image):
            run_pipeline(args, image_path=args.image, output_dir=args.output)
        else:
            logger.error("Image file not found.")
    else:
        run_batch(args)
