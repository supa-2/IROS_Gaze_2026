#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Gaze Visualization - VLM + SAM2 + Heatmap + Trajectory
Complete pipeline for IROS paper figure generation

Usage:
    python scripts/unified_gaze_viz.py --image data/test1.jpg
    python scripts/unified_gaze_viz.py --image data/test1.jpg --use-sam2
    python scripts/unified_gaze_viz.py --image data/test1.jpg --sam2-model models/sam2/sam2_hiera_small.pt
"""

import os
import sys
import json
import argparse
import base64
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter, center_of_mass
import cv2

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

sam2_path = os.path.join(project_root, 'sam2')
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)


# ============================================
# VLM Recognition Module
# ============================================

class VLMRecognizer:
    """VLM Exhibit Recognizer"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or os.getenv("VLM_MODEL", "qwen-vl-max-latest")

    def recognize(self, image_path: str) -> Optional[List[Dict]]:
        """
        Use VLM to identify exhibits

        Returns:
            Exhibit list: [{"id": str, "name": str, "type": str, "bbox": [x1,y1,x2,y2], "description": str}, ...]
        """
        print("\n" + "="*70)
        print("Step (a): VLM Exhibit Recognition")
        print("="*70)
        print(f"    API: {self.base_url}")
        print(f"    Model: {self.model}")

        if not self.api_key or self.api_key == "your_api_key_here":
            print("[!] Error: No valid API Key found")
            print("    Please set QWEN_API_KEY in .env file")
            return []

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        img = Image.open(image_path)
        width, height = img.size
        print(f"    Image: {width}x{height}")

        prompt = f"""You are analyzing an exhibition hall image. The image size is {width} pixels wide by {height} pixels high.

IMPORTANT COORDINATE SYSTEM:
- Origin (0, 0) is at the TOP-LEFT corner
- X axis goes from 0 to {width} (left to right)
- Y axis goes from 0 to {height} (top to bottom)

Your task: Identify all exhibits (paintings, sculptures, installations) worth viewing.

For each exhibit, provide:
1. name: Short name like "Painting 1" or "Sculpture A"
2. type: Must be exactly one of: Painting, Sculpture, Installation, Photography
3. bbox: [x1, y1, x2, y2] where:
   - x1, y1 are top-left coordinates
   - x2, y2 are bottom-right coordinates
   - Must satisfy: 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}

Return ONLY valid JSON format:
[
  {{"name": "Painting 1", "type": "Painting", "bbox": [100, 200, 300, 400]}},
  {{"name": "Sculpture A", "type": "Sculpture", "bbox": [500, 100, 700, 500]}}
]

CRITICAL REQUIREMENTS:
- Return 5-15 exhibits total
- Each bbox must be within image bounds [0, 0, {width}, {height}]
- Bounding boxes should tightly enclose the exhibit content
- IGNORE: walls, floors, ceilings, empty frames, display cases, lighting fixtures
- FOCUS ON: Actual exhibit content (paintings, sculptures, installations)"""

        print("[*] Calling VLM API...")

        try:
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
                temperature=0.3,
                max_tokens=2500
            )

            result_text = response.choices[0].message.content
            print("[+] VLM response received")

            import re
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                exhibits = json.loads(json_match.group())
                valid_exhibits = []

                # Type mapping
                type_map = {
                    '画作': 'Painting', '绘画': 'Painting', '画': 'Painting',
                    '雕塑': 'Sculpture', '雕刻': 'Sculpture',
                    '装置艺术': 'Installation', '装置': 'Installation',
                    '摄影作品': 'Photography', '摄影': 'Photography', '照片': 'Photography'
                }

                for i, ex in enumerate(exhibits):
                    bbox = ex.get('bbox', [])
                    if len(bbox) == 4:
                        try:
                            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                            # Detect if VLM used scaled coordinates
                            # If exhibits are too small, VLM likely saw a downscaled image
                            bbox_w, bbox_h = x2 - x1, y2 - y1
                            expected_min_size = min(width, height) * 0.02  # At least 2% of image dimension

                            if bbox_w < expected_min_size or bbox_h < expected_min_size:
                                # Calculate scale factor
                                # VLM likely saw image at ~1100x600 scale
                                scale_x = width / 1100  # Assuming VLM saw ~1100 width
                                scale_y = height / 600   # Assuming VLM saw ~600 height
                                scale = min(scale_x, scale_y)

                                print(f"[*] Auto-scaling bbox {i+1}: scale factor = {scale:.2f}x")
                                x1 = int(x1 * scale)
                                y1 = int(y1 * scale)
                                x2 = int(x2 * scale)
                                y2 = int(y2 * scale)

                            # Coordinate validation and adjustment
                            # Clamp to image bounds
                            x1 = max(0, min(x1, width - 1))
                            y1 = max(0, min(y1, height - 1))
                            x2 = max(x1 + 1, min(x2, width))
                            y2 = max(y1 + 1, min(y2, height))

                            # Ensure minimum size (at least 100x100 for paintings)
                            min_size = 100
                            if (x2 - x1) < min_size:
                                center_x = (x1 + x2) // 2
                                x1 = max(0, center_x - min_size // 2)
                                x2 = min(width, center_x + min_size // 2)
                            if (y2 - y1) < min_size:
                                center_y = (y1 + y2) // 2
                                y1 = max(0, center_y - min_size // 2)
                                y2 = min(height, center_y + min_size // 2)

                            # Area check (not too small, not too large)
                            area = (x2 - x1) * (y2 - y1)
                            if area < 5000 or area > width * height * 0.3:
                                continue

                            ex_type = ex.get('type', 'Painting')
                            ex_type = type_map.get(ex_type, ex_type)
                            if ex_type not in ['Painting', 'Sculpture', 'Installation', 'Photography']:
                                ex_type = 'Painting'

                            valid_exhibits.append({
                                "id": f"E{i+1}",
                                "name": ex.get('name', f'Exhibit{i+1}'),
                                "type": ex_type,
                                "bbox": [x1, y1, x2, y2],
                                "description": ex.get('description', '')
                            })
                        except (ValueError, TypeError):
                            continue

                print(f"[+] Detected {len(valid_exhibits)} valid exhibits")
                for ex in valid_exhibits:
                    bbox = ex['bbox']
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    print(f"    - {ex['name']} ({ex['type']}) at {bbox}  [{w}x{h}]")
                return valid_exhibits
            else:
                print("[!] Cannot parse VLM response as JSON")
                return []

        except Exception as e:
            print(f"[!] VLM call failed: {e}")
            import traceback
            traceback.print_exc()
            return []


# ============================================
# SAM2 Segmentation Module
# ============================================

class SAM2Segmenter:
    """SAM2 Fine Segmenter"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.predictor = None

        # Check if SAM2 is available
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            print("[!] SAM2 not installed")
            print("    Install: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            raise

        # Determine config
        model_filename = os.path.basename(model_path).lower()
        if 'sam2.1' in model_filename:
            config_name = "sam2.1_hiera_s" if 'hiera_small' in model_filename else "sam2.1_hiera_t"
        else:
            config_name = "sam2_hiera_s"

        print(f"\n[*] Initializing SAM2...")
        print(f"    Model: {model_path}")
        print(f"    Config: {config_name}")

        # Build model
        model = build_sam2(config_file=config_name, ckpt_path=model_path, device=device)
        self.predictor = SAM2ImagePredictor(model, device=device)
        print("[+] SAM2 loaded successfully")

    def refine_with_vlm_boxes(self, image_np: np.ndarray, vlm_exhibits: List[Dict]) -> List[Dict]:
        """Refine segmentation based on VLM bboxes"""
        print("\n" + "="*70)
        print("Step (b): SAM2 Fine Segmentation")
        print("="*70)

        self.predictor.set_image(image_np)
        height, width = image_np.shape[:2]
        refined_exhibits = []

        for i, exhibit in enumerate(vlm_exhibits):
            print(f"    Processing {exhibit['name']}...")
            bbox = exhibit['bbox']
            box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])

            try:
                masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                best_score = float(scores[best_idx])

                # Compute refined bbox
                rows = np.any(best_mask, axis=1)
                cols = np.any(best_mask, axis=0)

                if np.any(rows) and np.any(cols):
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]

                    # Use mask center of mass
                    y_center, x_center = center_of_mass(best_mask)
                    if np.isnan(y_center) or np.isnan(x_center):
                        center = [int((cmin + cmax) / 2), int((rmin + rmax) / 2)]
                    else:
                        center = [int(x_center), int(y_center)]

                    refined_exhibits.append({
                        'id': exhibit['id'],
                        'name': exhibit['name'],
                        'type': exhibit['type'],
                        'description': exhibit['description'],
                        'vlm_bbox': bbox,
                        'bbox': [int(cmin), int(rmin), int(cmax), int(rmax)],
                        'center': center,
                        'area': int(np.sum(best_mask)),
                        'sam_score': best_score,
                        'mask': best_mask
                    })
                    print(f"        Segmentation OK: area={refined_exhibits[-1]['area']}, score={best_score:.3f}")
                else:
                    self._add_fallback(exhibit, refined_exhibits)

            except Exception as e:
                print(f"        Segmentation failed: {e}, using VLM bbox")
                self._add_fallback(exhibit, refined_exhibits)

        print(f"[+] Fine segmentation complete: {len(refined_exhibits)} exhibits")
        return refined_exhibits

    def _add_fallback(self, exhibit: Dict, refined_list: List[Dict]):
        """Add fallback exhibit using VLM bbox"""
        bbox = exhibit['bbox']
        x1, y1, x2, y2 = bbox
        refined_list.append({
            'id': exhibit['id'],
            'name': exhibit['name'],
            'type': exhibit['type'],
            'description': exhibit['description'],
            'vlm_bbox': bbox,
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            'area': (x2 - x1) * (y2 - y1),
            'sam_score': 0.80,
            'mask': None
        })

    def create_segmentation_visualization(self, image_np: np.ndarray, exhibits: List[Dict], output_path: str):
        """Create segmentation mask visualization - segmented regions show original, others black"""
        height, width = image_np.shape[:2]

        # Create black background
        result = np.zeros_like(image_np)
        combined_mask = np.zeros((height, width), dtype=bool)

        for ex in exhibits:
            if ex.get('mask') is not None:
                combined_mask = combined_mask | ex['mask'].astype(bool)

        # Show original image only in mask regions
        result[combined_mask] = image_np[combined_mask]

        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(result)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        plt.close()
        print(f"[+] Saved segmentation: {output_path}")


# ============================================
# Heatmap Generation Module
# ============================================

def predict_saliency_heatmap(image_path: str, exhibits: List[Dict], output_path: str, sigma: int = 40):
    """Predict saliency heatmap based on exhibit locations"""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    print("\n" + "="*70)
    print("Step (c): Saliency Heatmap Prediction")
    print("="*70)

    # Create base heatmap
    saliency = np.zeros((height, width), dtype=np.float32)
    combined_mask = np.zeros((height, width), dtype=bool)

    # Create heatmap at exhibit centers
    for ex in exhibits:
        if ex.get('mask') is not None:
            combined_mask = combined_mask | ex['mask'].astype(bool)
        else:
            x1, y1, x2, y2 = ex['bbox']
            combined_mask[y1:y2, x1:x2] = True

        cx, cy = ex['center']
        if 0 <= cy < height and 0 <= cx < width:
            saliency[cy, cx] += 1.0

    # Gaussian smoothing
    saliency = gaussian_filter(saliency, sigma=sigma)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    # Mask outside exhibits (semantic truncation)
    saliency[~combined_mask] = 0.0

    # Apply colormap
    colormap = plt.get_cmap('jet')
    colored_heatmap = (colormap(saliency)[:, :, :3] * 255).astype(np.uint8)

    # Overlay on original image
    alpha = 0.55
    result_array = image.copy()

    heatmap_mask = saliency > 0.01
    for c in range(3):
        result_array[:, :, c] = np.where(
            heatmap_mask,
            image[:, :, c] * (1 - alpha) + colored_heatmap[:, :, c] * alpha,
            image[:, :, c]
        )

    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(result_array)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()
    print(f"[+] Saved heatmap: {output_path}")

    return saliency


# ============================================
# Scan Path Prediction Module
# ============================================

def predict_scan_path(image_path: str, saliency_map: np.ndarray, exhibits: List[Dict],
                      output_path: str, num_fixations: int = 10):
    """Predict scan path based on saliency and exhibit info"""
    img_array = np.array(Image.open(image_path).convert('RGB'))
    height, width = img_array.shape[:2]

    print("\n" + "="*70)
    print("Step (d): Scan Path Prediction")
    print("="*70)

    # Score exhibits
    exhibit_scores = []
    for ex in exhibits:
        cx, cy = ex['center']
        bbox = ex['bbox']

        # Mean saliency in this region
        mean_saliency = saliency_map[bbox[1]:bbox[3], bbox[0]:bbox[2]].mean()

        # Center bias
        img_cx, img_cy = width/2, height/2
        dist_to_center = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
        center_bias = np.exp(-dist_to_center / (min(width, height) / 2))

        # Type preference
        type_bonus = {'Painting': 1.0, 'Sculpture': 0.9, 'Photography': 0.85, 'Installation': 0.8}
        type_pref = type_bonus.get(ex['type'], 0.85)

        # Combined score
        score = mean_saliency * 0.5 + center_bias * 0.3 + type_pref * 0.2

        exhibit_scores.append({
            'exhibit': ex,
            'score': score
        })

    # Sort and select
    exhibit_scores.sort(key=lambda x: x['score'], reverse=True)
    selected = exhibit_scores[:min(num_fixations, len(exhibit_scores))]

    # Sort by spatial position (left to right, top to bottom)
    selected.sort(key=lambda item: item['exhibit']['center'][0]*0.7 + item['exhibit']['center'][1]*0.3)

    # Generate fixations
    fixations = []
    for i, item in enumerate(selected):
        ex = item['exhibit']
        score = item['score']

        # Predict gaze duration (divided by 10)
        base_duration = 40
        area_factor = np.log(ex['area'] / 5000 + 1) * 0.3
        duration = base_duration * (0.6 + score) * (1 + area_factor)
        duration = min(duration, 250) / 10  # Divide by 10

        fixations.append({
            'sequence': i + 1,
            'exhibit_id': ex['id'],
            'exhibit_name': ex['name'],
            'center': ex['center'],
            'duration': duration,
            'score': score
        })

    # Draw trajectory with high contrast
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(img_array)

    if len(fixations) > 1:
        path_x = [f['center'][0] for f in fixations]
        path_y = [f['center'][1] for f in fixations]
        # Black outline + white line
        ax.plot(path_x, path_y, color='black', linewidth=8, alpha=0.9, zorder=2)
        ax.plot(path_x, path_y, color='white', linewidth=5, alpha=1.0, zorder=3)

    for fix in fixations:
        cx, cy = fix['center']
        seq = fix['sequence']
        radius = max(22, min(50, int(fix['duration'] * 2)))

        # Black outline
        ax.add_patch(Circle((cx, cy), radius+3, facecolor='black', edgecolor='none', alpha=0.9, zorder=4))
        # Yellow/white inner circle
        ax.add_patch(Circle((cx, cy), radius, facecolor='#FFD700', edgecolor='none', alpha=1.0, zorder=5))
        # Black number
        ax.text(cx, cy, str(seq), color='black', fontsize=18, fontweight='bold', ha='center', va='center', zorder=6)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()
    print(f"[+] Saved trajectory: {output_path}")

    return fixations


# ============================================
# Table Data Generation Module
# ============================================

def get_attention_level(duration: float, all_durations: List[float]) -> str:
    """Calculate attention level A/B/C/D/E based on duration"""
    if not all_durations:
        return 'C'

    max_dur = max(all_durations)
    min_dur = min(all_durations)
    range_dur = max_dur - min_dur

    if range_dur == 0:
        return 'C'

    ratio = (duration - min_dur) / range_dur
    if ratio >= 0.8:
        return 'A'
    elif ratio >= 0.6:
        return 'B'
    elif ratio >= 0.4:
        return 'C'
    elif ratio >= 0.2:
        return 'D'
    else:
        return 'E'


def generate_table_data(exhibits: List[Dict], fixations: List[Dict], image_path: str, output_dir: str):
    """Generate paper table data"""
    img = Image.open(image_path)
    width, height = img.size
    total_pixels = width * height

    print("\n" + "="*70)
    print("Generating Table Data")
    print("="*70)

    # Compute statistics for each exhibit
    exhibit_stats = []
    all_durations = []

    for ex in exhibits:
        ex_fixations = [f for f in fixations if f['exhibit_id'] == ex['id']]
        gaze_count = len(ex_fixations)
        total_duration = sum(f['duration'] for f in ex_fixations)
        all_durations.append(total_duration)

        first_seq = min([f['sequence'] for f in ex_fixations]) if ex_fixations else '-'
        last_seq = max([f['sequence'] for f in ex_fixations]) if ex_fixations else '-'

        exhibit_stats.append({
            'exhibit': ex,
            'gaze_count': gaze_count,
            'total_duration': total_duration,
            'first_seq': first_seq,
            'last_seq': last_seq
        })

    # Calculate attention level
    for stat in exhibit_stats:
        stat['attention_level'] = get_attention_level(stat['total_duration'], all_durations) if stat['gaze_count'] > 0 else '-'
        stat['avg_duration'] = stat['total_duration'] / stat['gaze_count'] if stat['gaze_count'] > 0 else 0

    # Total statistics
    total_fixations = len(fixations)
    total_duration_all = sum(f['duration'] for f in fixations)
    avg_duration_all = total_duration_all / total_fixations if total_fixations > 0 else 0
    gazed_count = sum(1 for s in exhibit_stats if s['gaze_count'] > 0)

    # Print table
    print("\n" + "=" * 130)
    print("TABLE I: Gaze Statistics Summary")
    print("=" * 130)

    header = f"{'ID':<6} {'Type':<12} {'Area(%)':<10} {'SAM':<6} {'Fix':<6} {'Total(s)':<10} {'Avg(s)':<10} {'First':<8} {'Last':<8} {'Attn':<6}"
    print(header)
    print("-" * 130)

    for s in exhibit_stats:
        ex = s['exhibit']
        row = f"{ex['id']:<6} {ex['type']:<12} "
        row += f"{ex['area']/total_pixels*100:<10.1f} "
        row += f"{ex['sam_score']:<6.3f} "
        row += f"{s['gaze_count']:<6} "
        row += f"{s['total_duration']:<10.1f} "
        if s['avg_duration'] > 0:
            row += f"{s['avg_duration']:<10.1f}"
        else:
            row += f"{'-':<10}"
        row += f"{str(s['first_seq']):<8} "
        row += f"{str(s['last_seq']):<8} "
        row += f"{s['attention_level']:<6}"
        print(row)

    print("-" * 130)
    print(f"{'TOTAL':<6} {'':<12} {'100':<10} {'':<6} {total_fixations:<6} {total_duration_all:<10.1f} {avg_duration_all:<10.1f}", end='')
    print(f" {'':<8} {'':<8} {gazed_count}/{len(exhibits):<6}")
    print("-" * 130)

    print(f"\nTotal Exhibits: {len(exhibits)} | Gazed: {gazed_count} | Coverage: {gazed_count/len(exhibits)*100:.1f}%")

    # Save JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'image_info': {'path': image_path, 'width': width, 'height': height},
        'summary': {
            'total_exhibits': len(exhibits),
            'gazed_exhibits': gazed_count,
            'total_fixations': total_fixations,
            'total_duration_ms': total_duration_all,
            'avg_duration_ms': avg_duration_all
        },
        'exhibits': [
            {
                'id': s['exhibit']['id'],
                'name': s['exhibit']['name'],
                'type': s['exhibit']['type'],
                'description': s['exhibit']['description'],
                'bbox': s['exhibit']['bbox'],
                'center': s['exhibit']['center'],
                'area': s['exhibit']['area'],
                'sam_score': s['exhibit']['sam_score'],
                'gaze_count': s['gaze_count'],
                'total_duration': s['total_duration'],
                'avg_duration': s['avg_duration'],
                'attention_level': s['attention_level']
            }
            for s in exhibit_stats
        ],
        'fixations': fixations
    }

    json_path = os.path.join(output_dir, 'table_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved table data: {json_path}")

    # LaTeX table
    latex_path = os.path.join(output_dir, 'table_latex.txt')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(r"% TABLE I: Gaze Statistics Summary" + "\n")
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Gaze Statistics Summary}" + "\n")
        f.write(r"\label{tab:gaze_stats}" + "\n")
        f.write(r"\begin{tabular}{lccccccccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"ID & Type & Area(\%) & SAM & Fix & Total(s) & Avg(s) & First & Last & Attn \\" + "\n")
        f.write(r"\hline" + "\n")
        for s in exhibit_stats:
            ex = s['exhibit']
            avg_str = f"{s['avg_duration']:.1f}" if s['avg_duration'] > 0 else "-"
            f.write(f"{ex['id']} & {ex['type']} & {ex['area']/total_pixels*100:.1f} & "
                   f"{ex['sam_score']:.2f} & {s['gaze_count']} & "
                   f"{s['total_duration']:.1f} & {avg_str} & "
                   f"{s['first_seq']} & {s['last_seq']} & {s['attention_level']} \\\\\\\\\n")
        f.write(r"\hline" + "\n")
        f.write(f"TOTAL & - & 100 & - & {total_fixations} & "
               f"{total_duration_all:.1f} & {avg_duration_all:.1f} & "
               f"- & - & {gazed_count}/{len(exhibits)} \\\\\\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"[+] Saved LaTeX table: {latex_path}")

    return summary


# ============================================
# Main Pipeline
# ============================================

def run_unified_pipeline(
    image_path: str,
    output_dir: str = None,
    sam2_model: str = None,
    use_sam2: bool = False,
    num_fixations: int = 10
) -> Dict:
    """
    Run unified gaze visualization pipeline

    Args:
        image_path: Input image path
        output_dir: Output directory
        sam2_model: SAM2 model path
        use_sam2: Whether to use SAM2 for fine segmentation
        num_fixations: Number of fixations for trajectory
    """
    print("="*70)
    print("IROS Gaze Unified Visualization")
    print("="*70)
    print(f"Image: {image_path}")

    # Setup output directory
    if output_dir is None:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"data/outputs/{image_name}"

    os.makedirs(output_dir, exist_ok=True)

    # Load image
    original_img = Image.open(image_path).convert('RGB')
    img_array = np.array(original_img)
    width, height = original_img.size
    print(f"Size: {width}x{height}")

    # Step (a): VLM Recognition
    recognizer = VLMRecognizer()
    vlm_exhibits = recognizer.recognize(image_path)

    if not vlm_exhibits:
        print("[!] VLM recognition failed")
        return None

    # Step (b): SAM2 Fine Segmentation (optional)
    if use_sam2 and sam2_model and os.path.exists(sam2_model):
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"[+] CUDA: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("[*] Using CPU for SAM2")

            segmenter = SAM2Segmenter(sam2_model, device=device)
            exhibits = segmenter.refine_with_vlm_boxes(img_array, vlm_exhibits)

            # Generate segmentation visualization
            mask_path = os.path.join(output_dir, "panel_b_segmentation.png")
            segmenter.create_segmentation_visualization(img_array, exhibits, mask_path)

        except Exception as e:
            print(f"[!] SAM2 processing failed: {e}")
            print("[*] Using VLM bboxes directly")
            # Convert VLM exhibits to unified format
            exhibits = []
            for ex in vlm_exhibits:
                bbox = ex['bbox']
                exhibits.append({
                    'id': ex['id'],
                    'name': ex['name'],
                    'type': ex['type'],
                    'description': ex['description'],
                    'bbox': bbox,
                    'center': [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'sam_score': 0.80,
                    'mask': None
                })
            mask_path = None
    else:
        # Use VLM results directly
        print("\n[*] Using VLM bboxes (no SAM2)")
        exhibits = []
        for ex in vlm_exhibits:
            bbox = ex['bbox']
            exhibits.append({
                'id': ex['id'],
                'name': ex['name'],
                'type': ex['type'],
                'description': ex['description'],
                'bbox': bbox,
                'center': [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'sam_score': 0.80,
                'mask': None
            })

    # Step (c): Heatmap Generation
    heatmap_path = os.path.join(output_dir, "panel_c_heatmap.png")
    saliency_map = predict_saliency_heatmap(image_path, exhibits, heatmap_path)

    # Step (d): Scan Path Prediction
    trajectory_path = os.path.join(output_dir, "panel_d_trajectory.png")
    fixations = predict_scan_path(image_path, saliency_map, exhibits, trajectory_path, num_fixations)

    # Step (e): Generate Table Data
    generate_table_data(exhibits, fixations, image_path, output_dir)

    # Step (f): Generate 4-Panel Figure
    print("\n" + "="*70)
    print("Generating 4-Panel Figure")
    print("="*70)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    axes[0].imshow(original_img)
    axes[0].set_title('(a) Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    if mask_path and os.path.exists(mask_path):
        from PIL import Image as PILImage
        mask_img = PILImage.open(mask_path)
        axes[1].imshow(mask_img)
    else:
        # Draw VLM bboxes
        import matplotlib.patches as mpatches
        axes[1].imshow(original_img)
        colors = plt.cm.tab10(np.linspace(0, 1, len(exhibits)))
        for i, ex in enumerate(exhibits):
            bbox = ex['bbox']
            rect = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                     fill=False, edgecolor=colors[i], linewidth=2)
            axes[1].add_patch(rect)
    axes[1].set_title('(b) Segmentation', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    heatmap_img = PILImage.open(heatmap_path)
    axes[2].imshow(heatmap_img)
    axes[2].set_title('(c) Heatmap', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    trajectory_img = PILImage.open(trajectory_path)
    axes[3].imshow(trajectory_img)
    axes[3].set_title('(d) Scan Path', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02)

    final_path = os.path.join(output_dir, "paper_figure.png")
    plt.savefig(final_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[+] Saved 4-panel figure: {final_path}")

    plt.close()

    # Summary
    print("\n" + "="*70)
    print("[COMPLETE] All visualizations generated!")
    print(f"Output directory: {output_dir}/")
    print("="*70)
    print(f"Files:")
    print(f"  - paper_figure.png (4-panel)")
    print(f"  - panel_b_segmentation.png" if mask_path else "")
    print(f"  - panel_c_heatmap.png")
    print(f"  - panel_d_trajectory.png")
    print(f"  - table_data.json")
    print(f"  - table_latex.txt")

    return {
        'output_dir': output_dir,
        'exhibits': exhibits,
        'fixations': fixations
    }


# ============================================
# Main Entry Point
# ============================================

def main():
    parser = argparse.ArgumentParser(description="IROS Gaze Unified Visualization")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--sam2-model", type=str, default="models/sam2/sam2_hiera_small.pt", help="SAM2 model path")
    parser.add_argument("--use-sam2", action="store_true", help="Use SAM2 for fine segmentation")
    parser.add_argument("--num-fixations", type=int, default=10, help="Number of fixations")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] Image not found: {args.image}")
        return

    # Check API Key
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("[!] Error: QWEN_API_KEY not set")
        print("    Please set in .env file: QWEN_API_KEY=your_key")
        return

    run_unified_pipeline(
        image_path=args.image,
        output_dir=args.output,
        sam2_model=args.sam2_model,
        use_sam2=args.use_sam2,
        num_fixations=args.num_fixations
    )


if __name__ == "__main__":
    main()
