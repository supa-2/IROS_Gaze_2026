#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight video overlay pipeline for museum-gaze visualization.

Strategy:
1. Sample keyframes from the input video at a fixed interval.
2. Run the existing wall-art pipeline (VLM -> optional SAM2 -> heatmap -> scanpath) on sampled frames only.
3. Reproject / redraw the sampled-frame results back onto all video frames.

Outputs:
- overlay_heatmap.mp4
- overlay_scanpath.mp4
- overlay_segmentation.mp4
- keyframe_analysis.json
"""

import os
import sys
import json
import argparse
import base64
import time
import logging
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, center_of_mass
from contextlib import nullcontext

# Optional .env loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class VLMRecognizer:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or os.getenv("VLM_MODEL", "qwen-vl-max-latest")

    def recognize(self, image_path: str, max_retries: int = 3) -> List[Dict]:
        if not self.api_key:
            logger.error("No valid API Key found. Set QWEN_API_KEY in .env")
            return []

        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            img = Image.open(image_path)
            width, height = img.size
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
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        ],
                    }],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                result_text = response.choices[0].message.content
                exhibits = self._parse_and_validate_response(result_text, width, height)
                if exhibits:
                    return exhibits
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
                if ex_type not in {"Painting", "Photography"}:
                    continue
                area = max(1, (x2 - x1) * (y2 - y1))
                center_y = (y1 + y2) / 2
                if area < (width * height) * 0.0015:
                    continue
                if center_y > height * 0.88:
                    continue
                valid_exhibits.append({
                    "id": f"E{i + 1}",
                    "name": ex.get("name", f"Exhibit{i + 1}"),
                    "type": ex_type,
                    "bbox": [x1, y1, x2, y2],
                    "norm_bbox": bbox,
                    "description": ex.get("description", ""),
                })
            logger.info(f"Detected {len(valid_exhibits)} valid exhibits after filtering.")
            return valid_exhibits
        except Exception as e:
            logger.error(f"JSON parse failed: {e}")
            return []


class SAM2Segmenter:
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
            raise RuntimeError(f"Failed to load SAM2. Last error: {last_error}")

    def _infer_model_cfg_candidates(self, model_path: str) -> List[str]:
        model_name = os.path.basename(model_path).lower()
        primary = self._infer_primary_cfg(model_name)
        candidates = [primary]
        basename = os.path.basename(primary)
        if basename not in candidates:
            candidates.append(basename)
        if model_name.startswith("sam2.1"):
            alt = primary.replace("configs/sam2.1/", "")
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
    def _normalize_mask_output(masks) -> Optional[np.ndarray]:
        arr = np.asarray(masks)
        if arr.size == 0:
            return None
        if arr.ndim == 4:
            arr = arr[0, 0]
        elif arr.ndim == 3:
            arr = arr[0]
        else:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None
        return (arr > 0).astype(np.uint8)

    def refine_with_vlm_boxes(self, image_np: np.ndarray, vlm_exhibits: List[Dict]) -> List[Dict]:
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)
        self.predictor.set_image(image_np)
        refined = []
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
                    masks, scores, _ = self.predictor.predict(box=box_prompt, multimask_output=False)
                    mask = self._normalize_mask_output(masks)
                    if mask is None:
                        refined.append(self._fallback_entry(exhibit))
                        continue
                    y_idx, x_idx = np.where(mask > 0)
                    if len(x_idx) == 0 or len(y_idx) == 0:
                        refined.append(self._fallback_entry(exhibit))
                        continue
                    score_arr = np.asarray(scores)
                    score = float(score_arr.reshape(-1)[0]) if score_arr.size > 0 else 0.0
                    y1n, y2n = int(y_idx.min()), int(y_idx.max())
                    x1n, x2n = int(x_idx.min()), int(x_idx.max())
                    y_center, x_center = center_of_mass(mask)
                    refined.append({
                        **exhibit,
                        "bbox": [x1n, y1n, x2n, y2n],
                        "center": [int(x_center), int(y_center)],
                        "area": int(mask.sum()),
                        "sam_score": score,
                        "mask": mask.astype(bool),
                    })
                except Exception as e:
                    logger.warning(f"SAM2 failed for {exhibit['name']}: {e}")
                    refined.append(self._fallback_entry(exhibit))
        return refined

    def _fallback_entry(self, exhibit: Dict) -> Dict:
        x1, y1, x2, y2 = exhibit["bbox"]
        return {
            **exhibit,
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "area": max(1, (x2 - x1) * (y2 - y1)),
            "sam_score": 0.0,
            "mask": None,
        }


def vlm_fallback(vlm_exhibits: List[Dict]) -> List[Dict]:
    refined = []
    for ex in vlm_exhibits:
        bbox = ex["bbox"]
        refined.append({
            **ex,
            "center": [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)],
            "area": max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
            "sam_score": 0.8,
            "mask": None,
        })
    return refined


def generate_heatmap(image_np: np.ndarray, exhibits: List[Dict], sigma: int = 55) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_np.shape[:2]
    saliency = np.zeros((height, width), dtype=np.float32)
    for ex in exhibits:
        cx, cy = ex["center"]
        weight = 1.0 if ex.get("type", "Painting") == "Painting" else 0.95
        mask = ex.get("mask")
        if mask is not None:
            saliency += mask.astype(np.float32) * (0.08 * weight)
        if 0 <= cy < height and 0 <= cx < width:
            saliency[cy, cx] += 1.0 * weight
    saliency = gaussian_filter(saliency, sigma=sigma)
    if saliency.max() > 0:
        saliency /= saliency.max()
    heatmap_colored = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    result_img = cv2.addWeighted(image_np, 0.4, heatmap_colored, 0.6, 0)
    return saliency, result_img


def predict_scan_path(image_np: np.ndarray, saliency_map: np.ndarray, exhibits: List[Dict], num_fixations: int = 10) -> List[Dict]:
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
        type_bonus = 0.03 if ex.get("type", "Painting") == "Painting" else 0.02
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
            smooth_bonus = 0.0 if prev_dir is None else 0.10 * max(0.0, np.dot(delta / step_dist, prev_dir))
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
        path.append(chosen)

    fixations = []
    for i, item in enumerate(path):
        ex = item["exhibit"]
        duration = 1.2 + 2.8 * item["base_score"]
        duration += 0.25 * np.log1p(max(1, ex["area"]) / 5000.0)
        duration = float(np.clip(duration, 1.0, 6.5))
        fixations.append({
            "sequence": i + 1,
            "exhibit_id": ex["id"],
            "center": ex["center"],
            "duration": duration,
            "score": float(item["base_score"]),
        })
    return fixations


@dataclass
class KeyframeResult:
    frame_idx: int
    time_sec: float
    exhibits: List[Dict]
    fixations: List[Dict]
    saliency_map: np.ndarray


def save_temp_frame(frame_rgb: np.ndarray, temp_dir: str, frame_idx: int) -> str:
    path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
    Image.fromarray(frame_rgb).save(path, quality=95)
    return path


def analyze_frame(frame_rgb: np.ndarray, frame_idx: int, time_sec: float, args, vlm: VLMRecognizer, sam: Optional[SAM2Segmenter], temp_dir: str) -> Optional[KeyframeResult]:
    logger.info(f"Analyzing keyframe {frame_idx} at {time_sec:.2f}s")
    frame_path = save_temp_frame(frame_rgb, temp_dir, frame_idx)
    vlm_exhibits = vlm.recognize(frame_path)
    if not vlm_exhibits:
        logger.warning(f"No VLM exhibits for keyframe {frame_idx}")
        return None
    if args.use_sam2 and sam is not None:
        try:
            exhibits = sam.refine_with_vlm_boxes(frame_rgb, vlm_exhibits)
        except Exception as e:
            logger.warning(f"SAM2 failed on keyframe {frame_idx}: {e}")
            exhibits = vlm_fallback(vlm_exhibits)
    else:
        exhibits = vlm_fallback(vlm_exhibits)
    saliency_map, _ = generate_heatmap(frame_rgb, exhibits, sigma=args.heatmap_sigma)
    fixations = predict_scan_path(frame_rgb, saliency_map, exhibits, args.num_fixations)
    return KeyframeResult(frame_idx=frame_idx, time_sec=time_sec, exhibits=exhibits, fixations=fixations, saliency_map=saliency_map)


def serialize_keyframe_result(kf: KeyframeResult) -> Dict:
    return {
        "frame_idx": kf.frame_idx,
        "time_sec": kf.time_sec,
        "num_exhibits": len(kf.exhibits),
        "num_fixations": len(kf.fixations),
        "exhibits": [{k: v for k, v in ex.items() if k != "mask"} for ex in kf.exhibits],
        "fixations": kf.fixations,
    }


def blend_saliency(prev_map: np.ndarray, next_map: np.ndarray, alpha: float) -> np.ndarray:
    if prev_map is None and next_map is None:
        return None
    if prev_map is None:
        return next_map
    if next_map is None:
        return prev_map
    return (1.0 - alpha) * prev_map + alpha * next_map


def get_segment_keyframes(keyframes: List[KeyframeResult], frame_idx: int) -> Tuple[Optional[KeyframeResult], Optional[KeyframeResult], float]:
    if not keyframes:
        return None, None, 0.0
    if frame_idx <= keyframes[0].frame_idx:
        return keyframes[0], keyframes[0], 0.0
    if frame_idx >= keyframes[-1].frame_idx:
        return keyframes[-1], keyframes[-1], 0.0
    for i in range(len(keyframes) - 1):
        a = keyframes[i]
        b = keyframes[i + 1]
        if a.frame_idx <= frame_idx <= b.frame_idx:
            denom = max(1, b.frame_idx - a.frame_idx)
            alpha = (frame_idx - a.frame_idx) / denom
            return a, b, float(alpha)
    return keyframes[-1], keyframes[-1], 0.0


def render_heatmap_overlay(frame_rgb: np.ndarray, saliency_map: Optional[np.ndarray], alpha: float = 0.58) -> np.ndarray:
    if saliency_map is None:
        return frame_rgb.copy()
    sal = saliency_map.copy()
    if sal.max() > 0:
        sal = sal / sal.max()
    heatmap_bgr = cv2.applyColorMap((sal * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame_rgb, 1 - alpha, heatmap_rgb, alpha, 0)


def render_segmentation_overlay(frame_rgb: np.ndarray, exhibits: List[Dict], show_masks: bool = True) -> np.ndarray:
    out = frame_rgb.copy()
    colors = (plt_tab10_colors(max(1, len(exhibits))) * 255).astype(np.uint8)
    for i, ex in enumerate(exhibits):
        color = tuple(int(c) for c in colors[i % len(colors)])
        mask = ex.get("mask")
        if show_masks and mask is not None:
            overlay = out.copy()
            overlay[mask.astype(bool)] = color
            out = cv2.addWeighted(out, 0.78, overlay, 0.22, 0)
        x1, y1, x2, y2 = ex["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = ex["id"]
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def render_scanpath_overlay(frame_rgb: np.ndarray, fixations: List[Dict], progress: float) -> np.ndarray:
    out = frame_rgb.copy()
    if not fixations:
        return out
    n = len(fixations)
    visible = max(1, int(np.ceil(progress * n))) if n > 1 else 1
    visible_fix = fixations[:visible]
    pts = np.array([f["center"] for f in visible_fix], dtype=np.int32)
    if len(pts) > 1:
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (255, 255, 255), 3, cv2.LINE_AA)
    for fix in visible_fix:
        cx, cy = map(int, fix["center"])
        r = max(14, min(36, int(fix["duration"] * 5)))
        cv2.circle(out, (cx, cy), r + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r, (0, 215, 255), -1, cv2.LINE_AA)
        cv2.putText(out, str(fix["sequence"]), (cx - 8, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def plt_tab10_colors(n: int) -> np.ndarray:
    import matplotlib.pyplot as plt
    return plt.cm.tab10(np.linspace(0, 1, n))[:, :3]


def sample_keyframes(video_path: str, args) -> Tuple[List[KeyframeResult], Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    step = max(1, int(round(args.sample_every_sec * fps)))
    sample_indices = list(range(0, total_frames, step))
    if sample_indices[-1] != total_frames - 1:
        sample_indices.append(total_frames - 1)

    logger.info(f"Video FPS={fps:.3f}, frames={total_frames}, size={width}x{height}, duration={duration:.2f}s")
    logger.info(f"Sampling {len(sample_indices)} keyframes every {args.sample_every_sec:.2f}s ({step} frames)")

    vlm = VLMRecognizer()
    sam = None
    if args.use_sam2:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = SAM2Segmenter(args.sam2_model, device)

    keyframes: List[KeyframeResult] = []
    with tempfile.TemporaryDirectory(prefix="gaze_video_frames_") as temp_dir:
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok:
                logger.warning(f"Failed to read sampled frame {idx}")
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            kf = analyze_frame(frame_rgb, idx, idx / fps, args, vlm, sam, temp_dir)
            if kf is not None:
                keyframes.append(kf)
    cap.release()
    meta = {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": duration,
        "sample_indices": sample_indices,
    }
    return keyframes, meta


def write_overlay_videos(video_path: str, keyframes: List[KeyframeResult], meta: Dict, output_dir: str, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = meta["fps"]
    width = meta["width"]
    height = meta["height"]
    total_frames = meta["total_frames"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    heat_writer = cv2.VideoWriter(os.path.join(output_dir, "overlay_heatmap.mp4"), fourcc, fps, (width, height))
    seg_writer = cv2.VideoWriter(os.path.join(output_dir, "overlay_segmentation.mp4"), fourcc, fps, (width, height))
    scan_writer = cv2.VideoWriter(os.path.join(output_dir, "overlay_scanpath.mp4"), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        prev_kf, next_kf, alpha = get_segment_keyframes(keyframes, frame_idx)

        if prev_kf is None:
            heat_rgb = frame_rgb
            seg_rgb = frame_rgb
            scan_rgb = frame_rgb
        else:
            blended_sal = blend_saliency(prev_kf.saliency_map, next_kf.saliency_map, alpha)
            heat_rgb = render_heatmap_overlay(frame_rgb, blended_sal, alpha=args.video_heatmap_alpha)

            # segmentation follows nearest sampled analysis for stability
            seg_source = prev_kf if (next_kf is None or alpha < 0.5) else next_kf
            seg_rgb = render_segmentation_overlay(frame_rgb, seg_source.exhibits, show_masks=not args.box_only)

            # scanpath progressively reveals within the current segment
            if next_kf is not None and next_kf.frame_idx != prev_kf.frame_idx:
                progress = alpha
            else:
                progress = 1.0
            scan_rgb = render_scanpath_overlay(frame_rgb, prev_kf.fixations, progress=progress)

        heat_writer.write(cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))
        seg_writer.write(cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR))
        scan_writer.write(cv2.cvtColor(scan_rgb, cv2.COLOR_RGB2BGR))

        frame_idx += 1
        if frame_idx % max(1, int(fps)) == 0:
            logger.info(f"Rendered {frame_idx}/{total_frames} frames")

    cap.release()
    heat_writer.release()
    seg_writer.release()
    scan_writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--sample-every-sec", type=float, default=0.5, help="Analyze one frame every N seconds")
    parser.add_argument("--use-sam2", action="store_true")
    parser.add_argument("--sam2-model", type=str, default="models/sam2/sam2_hiera_small.pt")
    parser.add_argument("--num-fixations", type=int, default=8)
    parser.add_argument("--heatmap-sigma", type=int, default=55)
    parser.add_argument("--video-heatmap-alpha", type=float, default=0.58)
    parser.add_argument("--box-only", action="store_true", help="Draw only boxes in segmentation overlay")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    output_dir = args.output_dir or os.path.join(
        "data", "video_outputs", os.path.splitext(os.path.basename(args.video))[0]
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Input video: {args.video}")
    logger.info(f"Output dir: {output_dir}")

    start = time.time()
    keyframes, meta = sample_keyframes(args.video, args)
    if not keyframes:
        raise RuntimeError("No valid keyframes analyzed. VLM may have failed on all samples.")

    with open(os.path.join(output_dir, "keyframe_analysis.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "video": args.video,
            "meta": meta,
            "sampling_sec": args.sample_every_sec,
            "num_keyframes": len(keyframes),
            "keyframes": [serialize_keyframe_result(kf) for kf in keyframes],
        }, f, indent=2, ensure_ascii=False)

    write_overlay_videos(args.video, keyframes, meta, output_dir, args)
    logger.info(f"Done in {time.time() - start:.2f}s")
    logger.info(f"Saved: {os.path.join(output_dir, 'overlay_heatmap.mp4')}")
    logger.info(f"Saved: {os.path.join(output_dir, 'overlay_scanpath.mp4')}")
    logger.info(f"Saved: {os.path.join(output_dir, 'overlay_segmentation.mp4')}")


if __name__ == "__main__":
    main()
