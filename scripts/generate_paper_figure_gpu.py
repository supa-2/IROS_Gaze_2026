#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成论文用图表 - IROS Gaze 系统 (GPU服务器版本)
集成 VLM + SAM2 + 热力图 + 扫描路径
"""

import os
import sys
import argparse
import json
import base64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter
import torch
import cv2

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sam2_path = os.path.join(project_root, 'sam2')
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)


def call_qwen_vlm(image_path):
    """Use Qwen-VL to identify exhibits and locations"""
    print("\n" + "="*60)
    print("Step (a): VLM Exhibit Recognition")
    print("="*60)

    # Get API config
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("[!] Error: No valid API Key found")
        print("    Please set QWEN_API_KEY in .env file")
        return []

    base_url = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("VLM_MODEL", "qwen-vl-max-latest")

    print(f"    API: {base_url}")
    print(f"    Model: {model}")

    # Encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    prompt = """Analyze this exhibition hall image and identify all exhibits worth viewing.

For each exhibit, provide:
1. Name (concise, e.g., Painting 1, Sculpture A)
2. Type (must be one of: Painting, Sculpture, Installation, Photography)
3. Location in image (bounding box [x1, y1, x2, y2], where (0,0) is top-left)
4. Brief description (within 10 words)

Return in JSON format:
[
  {
    "name": "Exhibit Name",
    "type": "Painting/Sculpture/Installation/Photography",
    "bbox": [x1, y1, x2, y2],
    "description": "Description"
  }
]

Requirements:
- Only identify real exhibits, ignore walls, floors, display cases, lights
- Bounding box should tightly enclose the exhibit
- Return 5-12 main exhibits
- Type must be one of: Painting, Sculpture, Installation, Photography"""

    print("[*] Calling Qwen-VL API...")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content

        # Parse JSON
        import re
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            exhibits = json.loads(json_match.group())

            # Validate and filter
            valid_exhibits = []
            for ex in exhibits:
                bbox = ex.get('bbox', [])
                if len(bbox) == 4:
                    try:
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        area = (x2 - x1) * (y2 - y1)
                        if 500 < area < 600000:  # Reasonable area
                            # Normalize type
                            type_map = {
                                '画作': 'Painting', '绘画': 'Painting', '画': 'Painting',
                                '雕塑': 'Sculpture', '雕刻': 'Sculpture',
                                '装置艺术': 'Installation', '装置': 'Installation',
                                '摄影作品': 'Photography', '摄影': 'Photography', '照片': 'Photography'
                            }
                            ex_type = ex.get('type', 'Painting')
                            ex_type = type_map.get(ex_type, ex_type)
                            if ex_type not in ['Painting', 'Sculpture', 'Installation', 'Photography']:
                                ex_type = 'Painting'
                            valid_exhibits.append({
                                "name": ex.get('name', f'Exhibit{len(valid_exhibits)+1}'),
                                "type": ex_type,
                                "bbox": [x1, y1, x2, y2],
                                "description": ex.get('description', '')
                            })
                    except (ValueError, TypeError):
                        continue

            print(f"[+] VLM detected {len(valid_exhibits)} valid exhibits")
            return valid_exhibits
        else:
            print("[!] Cannot parse VLM response as JSON")
            return []

    except Exception as e:
        print(f"[!] VLM call failed: {e}")
        return []


class SAM2Segmenter:
    """SAM2 Fine Segmenter"""

    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device

        if not os.path.isabs(model_path):
            abs_model_path = os.path.join(project_root, model_path)
        else:
            abs_model_path = model_path

        print(f"\n[*] Initializing SAM2...")
        print(f"    Model: {model_path}")

        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(f"Model file not found: {abs_model_path}")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Determine config based on filename
            model_filename = os.path.basename(model_path).lower()
            if 'sam2.1' in model_filename:
                if 'hiera_small' in model_filename:
                    config_name = "sam2.1_hiera_s"
                elif 'hiera_tiny' in model_filename:
                    config_name = "sam2.1_hiera_t"
                else:
                    config_name = "sam2.1_hiera_s"
            else:
                if 'hiera_small' in model_filename or 'small' in model_filename:
                    config_name = "sam2_hiera_s"
                elif 'hiera_tiny' in model_filename or 'tiny' in model_filename:
                    config_name = "sam2_hiera_t"
                elif 'hiera_large' in model_filename or 'large' in model_filename:
                    config_name = "sam2_hiera_l"
                elif 'hiera_base+' in model_filename or 'b+' in model_filename:
                    config_name = "sam2_hiera_b+"
                else:
                    config_name = "sam2_hiera_s"

            print(f"    Config: {config_name}")

            model = build_sam2(
                config_file=config_name,
                ckpt_path=abs_model_path,
                device=device
            )

            self.predictor = SAM2ImagePredictor(model, device=device)
            print("[+] SAM2 loaded successfully")

        except Exception as e:
            print(f"[!] SAM2 loading failed: {e}")
            raise

    def refine_with_vlm_boxes(self, image_np, vlm_exhibits):
        """Refine segmentation based on VLM bboxes"""
        print("\n" + "="*60)
        print("Step (b): SAM2 Fine Segmentation")
        print("="*60)

        self.predictor.set_image(image_np)
        height, width = image_np.shape[:2]

        refined_exhibits = []

        for i, exhibit in enumerate(vlm_exhibits):
            print(f"    Processing {exhibit['name']}...")

            bbox = exhibit['bbox']
            x1, y1, x2, y2 = bbox
            box = np.array([x1, y1, x2, y2])

            try:
                masks, scores, logits = self.predictor.predict(
                    box=box,
                    multimask_output=True,
                )

                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                best_score = float(scores[best_idx])

                # Compute refined bbox
                rows = np.any(best_mask, axis=1)
                cols = np.any(best_mask, axis=0)

                if np.any(rows) and np.any(cols):
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]

                    refined_bbox = [int(cmin), int(rmin), int(cmax), int(rmax)]
                    center = [int((cmin + cmax) / 2), int((rmin + rmax) / 2)]
                    area = int((cmax - cmin) * (rmax - rmin))

                    refined_exhibits.append({
                        'id': f"E{i+1}",
                        'name': exhibit['name'],
                        'type': exhibit['type'],
                        'description': exhibit['description'],
                        'vlm_bbox': bbox,
                        'bbox': refined_bbox,
                        'center': center,
                        'area': area,
                        'sam_score': best_score,
                        'mask': best_mask
                    })
                    print(f"        Segmentation OK: area={area}, confidence={best_score:.3f}")
                else:
                    # Use original bbox
                    self._add_fallback(exhibit, i, refined_exhibits)

            except Exception as e:
                print(f"        Segmentation failed: {e}, using VLM bbox")
                self._add_fallback(exhibit, i, refined_exhibits)

        print(f"[+] Fine segmentation complete: {len(refined_exhibits)} exhibits")
        return refined_exhibits, image_np

    def _add_fallback(self, exhibit, idx, refined_list):
        """添加使用原始 VLM bbox 的展品"""
        bbox = exhibit['bbox']
        x1, y1, x2, y2 = bbox
        refined_list.append({
            'id': f"E{idx+1}",
            'name': exhibit['name'],
            'type': exhibit['type'],
            'description': exhibit['description'],
            'vlm_bbox': bbox,
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            'area': int((x2 - x1) * (y2 - y1)),
            'sam_score': 0.80,
            'mask': None
        })

    def create_segmentation_visualization(self, image_np, exhibits, output_path):
        """Create segmentation mask visualization"""
        height, width = image_np.shape[:2]

        # Create black background
        result = np.zeros_like(image_np)

        # Create combined mask
        combined_mask = np.zeros((height, width), dtype=bool)

        for ex in exhibits:
            if ex.get('mask') is not None:
                combined_mask = combined_mask | ex['mask']
            else:
                x1, y1, x2, y2 = ex['bbox']
                combined_mask[y1:y2, x1:x2] = True

        # Show original image only in mask regions
        result[combined_mask] = image_np[combined_mask]

        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(result)
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        plt.close()
        print(f"[+] Saved segmentation: {output_path}")

        return result


def predict_saliency_heatmap(image_path, exhibits, output_path, sigma=20):
    """Predict saliency heatmap based on exhibit locations"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    print("\n" + "="*60)
    print("Step (c): Saliency Prediction")
    print("="*60)

    # 创建基础显著性图
    saliency = np.zeros((height, width), dtype=np.float32)

    for ex in exhibits:
        bbox = ex['bbox']
        x1, y1, x2, y2 = bbox
        cx, cy = ex['center']

        # 基于面积的基础显著性
        area_ratio = ex['area'] / (width * height)
        base_saliency = 0.5 + min(0.5, area_ratio * 10)

        # 在 bbox 区域创建高斯分布
        y, x = np.mgrid[y1:y2, x1:x2]
        if y.size > 0 and x.size > 0:
            local_sigma = min(x2-x1, y2-y1) / 4
            if local_sigma > 1:
                gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * local_sigma**2))
                # 只在有效范围内设置
                valid_y = np.clip(y, 0, height-1).astype(int)
                valid_x = np.clip(x, 0, width-1).astype(int)
                for iy, ix, val in zip(valid_y.flatten(), valid_x.flatten(), gaussian.flatten()):
                    if 0 <= iy < height and 0 <= ix < width:
                        saliency[iy, ix] = max(saliency[iy, ix], val * base_saliency)

    # 添加中心偏置
    cy, cx = height // 2, width // 2
    y, x = np.mgrid[:height, :width]
    center_bias = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (min(height, width) / 2.5)**2))
    saliency = saliency * 0.7 + center_bias * 0.3

    # 平滑和归一化
    saliency = gaussian_filter(saliency, sigma=sigma)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    # 增强对比度
    saliency = np.power(saliency, 0.4)

    # 使用 'jet' 色图
    colormap = plt.get_cmap('jet')
    colored_heatmap = colormap(saliency)

    # 叠加到原图
    alpha = 0.45
    result_array = image.copy().astype(np.float32)

    mask = saliency > 0.02
    for c in range(3):
        result_array[:, :, c] = (
            image[:, :, c] * alpha +
            colored_heatmap[:, :, c] * 255 * (1 - alpha)
        )

    result_array = np.clip(result_array, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(result_array)
    ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] Saved heatmap: {output_path}")

    return saliency


def predict_scan_path(image_path, saliency_map, exhibits, output_path, num_fixations=10):
    """Predict scan path based on saliency and exhibit info"""
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    print("\n" + "="*60)
    print("Step (d): Scan Path Prediction")
    print("="*60)

    # Compute gaze score for each exhibit
    exhibit_scores = []
    for ex in exhibits:
        bbox = ex['bbox']
        x1, y1, x2, y2 = bbox
        cx, cy = ex['center']

        # Mean saliency in this region
        mean_saliency = saliency_map[y1:y2, x1:x2].mean() if y2 > y1 and x2 > x1 else 0

        # Center bias
        img_cx, img_cy = width/2, height/2
        dist_to_center = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
        center_bias = np.exp(-dist_to_center / (min(width, height) / 2))

        # Type preference: Painting > Sculpture > others
        type_bonus = {'Painting': 1.0, 'Sculpture': 0.9, 'Photography': 0.85, 'Installation': 0.8}
        type_pref = type_bonus.get(ex['type'], 0.85)

        # Combined score
        score = (mean_saliency * 0.5 + center_bias * 0.3 + type_pref * 0.2)

        exhibit_scores.append({
            'exhibit': ex,
            'score': score,
            'mean_saliency': mean_saliency
        })

    # Sort by score, select top N
    exhibit_scores.sort(key=lambda x: x['score'], reverse=True)
    selected = exhibit_scores[:min(num_fixations, len(exhibit_scores))]

    # Sort by spatial position (left to right, top to bottom)
    def scan_order_key(item):
        cx, cy = item['exhibit']['center']
        return cx * 0.6 + cy * 0.4

    selected.sort(key=scan_order_key)

    # Generate fixation data
    fixations = []
    for i, item in enumerate(selected):
        ex = item['exhibit']
        score = item['score']

        # Predict gaze duration (based on score and area), divided by 10
        base_duration = 40
        area_factor = np.log(ex['area'] / 5000 + 1) * 0.3
        duration = base_duration * (0.6 + score) * (1 + area_factor)
        duration = min(duration, 250) / 10  # Divide by 10!

        fixations.append({
            'sequence': i + 1,
            'exhibit_id': ex['id'],
            'exhibit_name': ex['name'],
            'center': ex['center'],
            'duration': duration,
            'score': score
        })

    # 绘制
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(img_array)

    # 路径线
    if len(fixations) > 1:
        path_x = [f['center'][0] for f in fixations]
        path_y = [f['center'][1] for f in fixations]
        ax.plot(path_x, path_y, color='black', linewidth=10, alpha=0.85, zorder=2)
        ax.plot(path_x, path_y, color='white', linewidth=6, alpha=1.0, zorder=3)

    # 注视点
    for fix in fixations:
        cx, cy = fix['center']
        duration = fix['duration']
        seq = fix['sequence']

        radius = max(28, min(65, int(duration / 3.5)))

        circle = Circle((cx, cy), radius, facecolor='white',
                       edgecolor='black', linewidth=7, alpha=0.95, zorder=4)
        ax.add_patch(circle)

        circle_inner = Circle((cx, cy), radius - 4, facecolor='white',
                       edgecolor='white', linewidth=4, alpha=0.9, zorder=5)
        ax.add_patch(circle_inner)

        ax.text(cx, cy, str(seq), color='black', fontsize=20, fontweight='bold',
               ha='center', va='center', zorder=6)

    ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] Saved trajectory: {output_path}")

    return fixations


def get_attention_level(duration, all_durations):
    """Calculate attention level A/B/C/D/E based on duration"""
    if not all_durations:
        return 'C'

    max_dur = max(all_durations)
    min_dur = min(all_durations)
    range_dur = max_dur - min_dur

    if range_dur == 0:
        return 'C'

    # 5 levels
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


def generate_table_data(exhibits, fixations, image_path, output_dir):
    """Generate paper table data"""
    img = Image.open(image_path)
    width, height = img.size
    total_pixels = width * height

    print("\n" + "="*60)
    print("Generating Table Data")
    print("="*60)

    # Compute gaze statistics for each exhibit
    exhibit_stats = []
    all_durations = []

    for ex in exhibits:
        # Find fixations for this exhibit
        ex_fixations = [f for f in fixations if f['exhibit_id'] == ex['id']]
        gaze_count = len(ex_fixations)
        total_duration = sum(f['duration'] for f in ex_fixations)
        all_durations.append(total_duration)

        # First and last fixation
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

    # Compute total statistics
    total_fixations = len(fixations)
    total_duration_all = sum(f['duration'] for f in fixations)
    avg_duration_all = total_duration_all / total_fixations if total_fixations > 0 else 0
    gazed_count = sum(1 for s in exhibit_stats if s['gaze_count'] > 0)

    # Print table
    print("\n" + "=" * 140)
    print("TABLE I: Gaze Statistics Summary")
    print("=" * 140)

    # 表头
    header = f"{'ID':<6} {'Type':<12} {'Area(%)':<10} {'SAM':<6} {'Fix':<6} {'Total(s)':<10} {'Avg(s)':<10} {'First':<8} {'Last':<8} {'Attn':<6} {'Center':<12}"
    print(header)
    print("-" * 140)

    for stat in exhibit_stats:
        ex = stat['exhibit']
        row = f"{ex['id']:<6} {ex['type']:<12} "
        row += f"{ex['area']/total_pixels*100:<10.1f} "
        row += f"{ex['sam_score']:<6.3f} "
        row += f"{stat['gaze_count']:<6} "
        row += f"{stat['total_duration']/1000:<10.1f} "
        if stat['avg_duration'] > 0:
            row += f"{stat['avg_duration']/1000:<10.1f} "
        else:
            row += f"{'-':<10} "
        row += f"{str(stat['first_seq']):<8} "
        row += f"{str(stat['last_seq']):<8} "
        row += f"{stat['attention_level']:<6} "
        row += f"({ex['center'][0]},{ex['center'][1]})"
        print(row)

    print("-" * 140)
    print(f"{'TOTAL':<6} {'':<12} {'100':<10} {'':<6} {total_fixations:<6} {total_duration_all/1000:<10.1f} {avg_duration_all/1000:<10.1f}", end='')
    print(f" {'':<8} {'':<8} {gazed_count}/{len(exhibits):<6}")
    print("-" * 140)

    print(f"\nTotal Exhibits: {len(exhibits)} | Gazed: {gazed_count} | Coverage: {gazed_count/len(exhibits)*100:.1f}%")

    # Save JSON
    summary = {
        'timestamp': '2026-03-03T00:00:00',
        'image_info': {
            'path': image_path,
            'width': width,
            'height': height
        },
        'summary': {
            'total_exhibits': len(exhibits),
            'gazed_exhibits': gazed_count,
            'total_fixations': total_fixations,
            'total_duration_ms': total_duration_all,
            'avg_duration_ms': avg_duration_all,
            'coverage_percent': gazed_count/len(exhibits)*100 if exhibits else 0
        },
        'exhibits': [
            {
                'id': s['exhibit']['id'],
                'name': s['exhibit']['name'],
                'type': s['exhibit']['type'],
                'description': s['exhibit']['description'],
                'bbox': s['exhibit']['bbox'],
                'center': s['exhibit']['center'],
                'area_pixels': s['exhibit']['area'],
                'area_percent': s['exhibit']['area']/total_pixels*100,
                'sam_score': s['exhibit']['sam_score'],
                'gaze_count': s['gaze_count'],
                'total_duration_ms': s['total_duration'],
                'avg_duration_ms': s['avg_duration'],
                'first_fixation': s['first_seq'],
                'last_fixation': s['last_seq'],
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
        f.write(r"\begin{tabular}{lcccccccccc}" + "\n")
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


def create_paper_figure(image_path, output_path, sam2_model_path, num_fixations=10):
    """Generate paper figure with 4 panels"""

    if not torch.cuda.is_available():
        print("[!] CUDA not available")
        return False

    print(f"\n{'='*60}")
    print(f"IROS Gaze Paper Figure Generation")
    print(f"{'='*60}")
    print(f"Image: {image_path}")

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    original_img = Image.open(image_path).convert('RGB')
    img_array = np.array(original_img)
    width, height = original_img.size
    print(f"Size: {width}x{height}")

    # Step 1: VLM recognition
    vlm_exhibits = call_qwen_vlm(image_path)
    if not vlm_exhibits:
        print("[!] VLM recognition failed, exiting")
        return False

    # Step 2: SAM2 fine segmentation
    segmenter = SAM2Segmenter(sam2_model_path)
    exhibits, _ = segmenter.refine_with_vlm_boxes(img_array, vlm_exhibits)

    mask_path = output_path.replace('.png', '_mask.png')
    segmenter.create_segmentation_visualization(img_array, exhibits, mask_path)

    # Step 3: Heatmap
    heatmap_path = output_path.replace('.png', '_heatmap.png')
    saliency_map = predict_saliency_heatmap(image_path, exhibits, heatmap_path)

    # Step 4: Scan path
    trajectory_path = output_path.replace('.png', '_trajectory.png')
    fixations = predict_scan_path(image_path, saliency_map, exhibits, trajectory_path, num_fixations)

    # Step 5: Generate table data
    generate_table_data(exhibits, fixations, image_path, output_dir)

    # Step 6: Generate 4-panel figure
    print("\n" + "="*60)
    print("Generating 4-Panel Figure")
    print("="*60)

    mask_img = Image.open(mask_path)
    heatmap_img = Image.open(heatmap_path)
    trajectory_img = Image.open(trajectory_path)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    axes[0].imshow(original_img)
    axes[0].set_title('(a) Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask_img)
    axes[1].set_title('(b) Segmentation', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(heatmap_img)
    axes[2].set_title('(c) Heatmap', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(trajectory_img)
    axes[3].set_title('(d) Scan Path', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[+] Saved 4-panel figure: {output_path}")

    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/outputs/paper_figure.png')
    parser.add_argument('--sam2-model', type=str, default='models/sam2/sam2_hiera_small.pt')
    parser.add_argument('--num-fixations', type=int, default=10)

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

    if torch.cuda.is_available():
        print(f"[+] CUDA: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("[!] CUDA not available")
        return

    create_paper_figure(
        args.image, args.output, args.sam2_model, args.num_fixations
    )

    print("\n[OK] Done!")


if __name__ == '__main__':
    main()
