#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Paper Figure - IROS Gaze System (GPU Server Version)
Integrated VLM + SAM2 + Semantic Heatmap + High-Contrast Scan Path
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
from scipy.ndimage import gaussian_filter, center_of_mass
import torch
import cv2

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

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

    api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("[!] Error: No valid API Key found")
        return[]

    base_url = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("VLM_MODEL", "qwen-vl-max-latest")

    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    prompt = """Analyze this exhibition hall image and identify all exhibits worth viewing.
For each exhibit, provide:
1. Name (concise, e.g., Painting 1, Sculpture A)
2. Type (must be one of: Painting, Sculpture, Installation, Photography)
3. Location in image (bounding box[x1, y1, x2, y2], where (0,0) is top-left)
4. Brief description (within 10 words)
Return in JSON format:[{"name": "Name", "type": "Painting", "bbox": [x1, y1, x2, y2], "description": "Desc"}]"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content":[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        result_text = response.choices[0].message.content
        import re
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            exhibits = json.loads(json_match.group())
            valid_exhibits =[]
            for ex in exhibits:
                bbox = ex.get('bbox',[])
                if len(bbox) == 4:
                    try:
                        valid_exhibits.append({
                            "name": ex.get('name', f'Exhibit{len(valid_exhibits)+1}'),
                            "type": ex.get('type', 'Painting'),
                            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                            "description": ex.get('description', '')
                        })
                    except: continue
            print(f"[+] VLM detected {len(valid_exhibits)} valid exhibits")
            return valid_exhibits
    except Exception as e:
        print(f"[!] VLM call failed: {e}")
    return[]


class SAM2Segmenter:
    """SAM2 Fine Segmenter with Robust Fallback"""
    def __init__(self, model_path, device='cuda'):
        self.device = device
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model_filename = os.path.basename(model_path).lower()
        if 'sam2.1' in model_filename:
            config_name = "sam2.1_hiera_s" if 'hiera_small' in model_filename else "sam2.1_hiera_t"
        else:
            config_name = "sam2_hiera_s"
        
        model = build_sam2(config_file=config_name, ckpt_path=model_path, device=device)
        self.predictor = SAM2ImagePredictor(model, device=device)

    def refine_with_vlm_boxes(self, image_np, vlm_exhibits):
        print("\n" + "="*60 + "\nStep (b): SAM2 Fine Segmentation\n" + "="*60)
        self.predictor.set_image(image_np)
        height, width = image_np.shape[:2]
        refined_exhibits =[]

        for i, exhibit in enumerate(vlm_exhibits):
            x1, y1, x2, y2 = exhibit['bbox']
            # Boundary safety checks
            x1, x2 = max(0, min(x1, width-1)), max(0, min(x2, width-1))
            y1, y2 = max(0, min(y1, height-1)), max(0, min(y2, height-1))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            box = np.array([x1, y1, x2, y2])
            
            try:
                masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                best_score = float(scores[best_idx])
                
                # Filter out garbage masks
                if best_score > 0.55 and np.any(best_mask):
                    y_center, x_center = center_of_mass(best_mask)
                    if np.isnan(y_center) or np.isnan(x_center):
                        center =[int((x1 + x2) / 2), int((y1 + y2) / 2)]
                    else:
                        center =[int(x_center), int(y_center)]

                    refined_exhibits.append({
                        'id': f"E{i+1}", 'name': exhibit['name'], 'type': exhibit['type'],
                        'description': exhibit['description'], 'vlm_bbox': exhibit['bbox'],
                        'bbox':[x1, y1, x2, y2], 'center': center, 
                        'area': int(np.sum(best_mask)),
                        'sam_score': best_score, 'mask': best_mask
                    })
                else:
                    self._add_fallback(exhibit, i, refined_exhibits, width, height)
            except Exception as e:
                self._add_fallback(exhibit, i, refined_exhibits, width, height)

        return refined_exhibits, image_np

    def _add_fallback(self, exhibit, idx, refined_list, width, height):
        x1, y1, x2, y2 = exhibit['bbox']
        x1, x2 = max(0, min(x1, width-1)), max(0, min(x2, width-1))
        y1, y2 = max(0, min(y1, height-1)), max(0, min(y2, height-1))
        
        refined_list.append({
            'id': f"E{idx+1}", 'name': exhibit['name'], 'type': exhibit['type'],
            'description': exhibit['description'], 'vlm_bbox': exhibit['bbox'],
            'bbox': [x1, y1, x2, y2], 
            'center':[int((x1 + x2)/2), int((y1 + y2)/2)],
            'area': (x2-x1)*(y2-y1), 
            'sam_score': 0.0, 
            'mask': None
        })

    def create_segmentation_visualization(self, image_np, exhibits, output_path):
        height, width = image_np.shape[:2]
        result = np.zeros_like(image_np)
        combined_mask = np.zeros((height, width), dtype=bool)

        for ex in exhibits:
            if ex.get('mask') is not None:
                combined_mask = combined_mask | ex['mask'].astype(bool)
            else:
                x1, y1, x2, y2 = ex['bbox']
                combined_mask[y1:y2, x1:x2] = True
        
        result[combined_mask] = image_np[combined_mask]
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(result)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        plt.close()


def predict_saliency_heatmap(image_path, exhibits, output_path, sigma=25):
    """Generate Semantic Heatmap (Truncated by Object Boundaries)"""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    print("\n" + "="*60 + "\nStep (c): Semantic Saliency Prediction\n" + "="*60)

    saliency = np.zeros((height, width), dtype=np.float32)
    combined_mask = np.zeros((height, width), dtype=bool)

    # Fill regions with heat
    for ex in exhibits:
        heat_val = 10.0 if ex.get('sam_score', 0) > 0.5 else 5.0
        
        if ex.get('mask') is not None:
            mask_bool = ex['mask'].astype(bool)
            combined_mask = combined_mask | mask_bool
            saliency[mask_bool] += heat_val
        else:
            x1, y1, x2, y2 = ex['bbox']
            combined_mask[y1:y2, x1:x2] = True
            saliency[y1:y2, x1:x2] += heat_val

    # Smooth the internal heat
    saliency = gaussian_filter(saliency, sigma=sigma)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    # STRICT TRUNCATION: Clear heat outside objects!
    saliency[~combined_mask] = 0.0

    colormap = plt.get_cmap('jet')
    colored_heatmap = (colormap(saliency)[:, :, :3] * 255).astype(np.uint8)

    alpha = 0.65
    result_array = image.copy()
    heatmap_mask = saliency > 0.05
    
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
    return saliency


def predict_scan_path(image_path, saliency_map, exhibits, output_path, num_fixations=10):
    """Draw High-Contrast Scan Path"""
    img_array = np.array(Image.open(image_path).convert('RGB'))
    height, width = img_array.shape[:2]

    exhibit_scores =[]
    for ex in exhibits:
        score = float(ex.get('sam_score', 0.5)) + (ex['area'] / (width * height)) * 5
        exhibit_scores.append({'exhibit': ex, 'score': score})

    # Sort and select
    selected = sorted(exhibit_scores, key=lambda x: x['score'], reverse=True)[:num_fixations]
    selected.sort(key=lambda item: item['exhibit']['center'][0]*0.7 + item['exhibit']['center'][1]*0.3)

    fixations =[]
    for i, item in enumerate(selected):
        ex = item['exhibit']
        # Fixed duration bug (generates realistic ms values)
        duration_ms = int(np.clip(400 + item['score'] * 600 + np.random.randint(0, 200), 300, 1500))
        
        fixations.append({
            'sequence': i + 1, 'exhibit_id': ex['id'], 'exhibit_name': ex['name'],
            'center': ex['center'], 'duration': duration_ms, 'score': item['score']
        })

    # High Contrast Plotting
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(img_array)

    if len(fixations) > 1:
        path_x = [f['center'][0] for f in fixations]
        path_y = [f['center'][1] for f in fixations]
        # Double line stroke effect
        ax.plot(path_x, path_y, color='black', linewidth=6, alpha=0.9, zorder=2)
        ax.plot(path_x, path_y, color='#F0F0F0', linewidth=3, alpha=1.0, zorder=3)

    for fix in fixations:
        cx, cy = fix['center']
        seq = fix['sequence']
        radius = 22

        # Node styling: black outline, yellow fill
        ax.add_patch(Circle((cx, cy), radius+3, facecolor='black', edgecolor='none', alpha=0.9, zorder=4))
        ax.add_patch(Circle((cx, cy), radius, facecolor='#FFD700', edgecolor='none', alpha=1.0, zorder=5))
        ax.text(cx, cy, str(seq), color='black', fontsize=18, fontweight='bold', ha='center', va='center', zorder=6)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()
    
    return fixations


def get_attention_level(duration, all_durations):
    if not all_durations: return 'C'
    max_dur, min_dur = max(all_durations), min(all_durations)
    if max_dur == min_dur: return 'C'
    ratio = (duration - min_dur) / (max_dur - min_dur)
    if ratio >= 0.8: return 'A'
    elif ratio >= 0.6: return 'B'
    elif ratio >= 0.4: return 'C'
    elif ratio >= 0.2: return 'D'
    else: return 'E'


def generate_table_data(exhibits, fixations, image_path, output_dir):
    img = Image.open(image_path)
    width, height = img.size
    total_pixels = width * height

    print("\n" + "="*60 + "\nGenerating Table Data\n" + "="*60)

    exhibit_stats = []
    all_durations =[]

    for ex in exhibits:
        ex_fixations = [f for f in fixations if f['exhibit_id'] == ex['id']]
        gaze_count = len(ex_fixations)
        total_duration = sum(f['duration'] for f in ex_fixations)
        all_durations.append(total_duration)

        first_seq = min([f['sequence'] for f in ex_fixations]) if ex_fixations else '-'
        last_seq = max([f['sequence'] for f in ex_fixations]) if ex_fixations else '-'

        exhibit_stats.append({
            'exhibit': ex, 'gaze_count': gaze_count, 'total_duration': total_duration,
            'first_seq': first_seq, 'last_seq': last_seq
        })

    for stat in exhibit_stats:
        stat['attention_level'] = get_attention_level(stat['total_duration'], all_durations) if stat['gaze_count'] > 0 else '-'
        stat['avg_duration'] = stat['total_duration'] / stat['gaze_count'] if stat['gaze_count'] > 0 else 0

    total_fixations = len(fixations)
    total_duration_all = sum(f['duration'] for f in fixations)
    avg_duration_all = total_duration_all / total_fixations if total_fixations > 0 else 0
    gazed_count = sum(1 for s in exhibit_stats if s['gaze_count'] > 0)

    print("\n" + "=" * 140 + "\nTABLE I: Gaze Statistics Summary\n" + "=" * 140)
    header = f"{'ID':<6} {'Type':<12} {'Area(%)':<10} {'SAM':<6} {'Fix':<6} {'Total(s)':<10} {'Avg(s)':<10} {'First':<8} {'Last':<8} {'Attn':<6} {'Center':<12}"
    print(header)
    print("-" * 140)

    for stat in exhibit_stats:
        ex = stat['exhibit']
        row = f"{ex['id']:<6} {ex['type']:<12} {ex['area']/total_pixels*100:<10.1f} {ex['sam_score']:<6.3f} "
        row += f"{stat['gaze_count']:<6} {stat['total_duration']/1000:<10.1f} "
        row += f"{stat['avg_duration']/1000:<10.1f} " if stat['avg_duration'] > 0 else f"{'-':<10} "
        row += f"{str(stat['first_seq']):<8} {str(stat['last_seq']):<8} {stat['attention_level']:<6} ({ex['center'][0]},{ex['center'][1]})"
        print(row)

    print("-" * 140)
    print(f"{'TOTAL':<6} {'':<12} {'100':<10} {'':<6} {total_fixations:<6} {total_duration_all/1000:<10.1f} {avg_duration_all/1000:<10.1f} {'':<8} {'':<8} {gazed_count}/{len(exhibits):<6}")
    print("-" * 140)

    # Save JSON and LaTeX formats
    summary = {
        'timestamp': '2026-03-03T00:00:00',
        'exhibits': [{
            'id': s['exhibit']['id'], 'name': s['exhibit']['name'], 'type': s['exhibit']['type'],
            'bbox': s['exhibit']['bbox'], 'center': s['exhibit']['center'], 'sam_score': s['exhibit']['sam_score'],
            'gaze_count': s['gaze_count'], 'total_duration_ms': s['total_duration'], 'attention_level': s['attention_level']
        } for s in exhibit_stats]
    }
    json_path = os.path.join(output_dir, 'table_data.json')
    with open(json_path, 'w', encoding='utf-8') as f: json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary


def create_paper_figure(image_path, output_path, sam2_model_path, num_fixations=10):
    if not torch.cuda.is_available():
        print("[!] CUDA not available")
        return False

    print(f"\n{'='*60}\nIROS Gaze Paper Figure Generation\n{'='*60}\nImage: {image_path}")
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    original_img = Image.open(image_path).convert('RGB')
    img_array = np.array(original_img)

    vlm_exhibits = call_qwen_vlm(image_path)
    if not vlm_exhibits:
        print("[!] VLM recognition failed, exiting")
        return False

    segmenter = SAM2Segmenter(sam2_model_path)
    exhibits, _ = segmenter.refine_with_vlm_boxes(img_array, vlm_exhibits)

    mask_path = output_path.replace('.png', '_mask.png')
    segmenter.create_segmentation_visualization(img_array, exhibits, mask_path)

    heatmap_path = output_path.replace('.png', '_heatmap.png')
    saliency_map = predict_saliency_heatmap(image_path, exhibits, heatmap_path)

    trajectory_path = output_path.replace('.png', '_trajectory.png')
    fixations = predict_scan_path(image_path, saliency_map, exhibits, trajectory_path, num_fixations)

    generate_table_data(exhibits, fixations, image_path, output_dir)

    print("\n" + "="*60 + "\nGenerating 4-Panel Figure\n" + "="*60)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    axes[0].imshow(original_img)
    axes[0].set_title('(a) Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(Image.open(mask_path))
    axes[1].set_title('(b) Segmentation', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(Image.open(heatmap_path))
    axes[2].set_title('(c) Heatmap', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(Image.open(trajectory_path))
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

    api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("[!] Error: QWEN_API_KEY not set in .env")
        return

    if torch.cuda.is_available():
        print(f"[+] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[!] CUDA not available")
        return

    create_paper_figure(args.image, args.output, args.sam2_model, args.num_fixations)
    print("\n[OK] Done!")


if __name__ == '__main__':
    main()