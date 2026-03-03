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
    #[保持你原有的逻辑不变]
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
    def __init__(self, model_path, device='cuda'):
        self.device = device
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        # 自动推断 config (保持你原有逻辑)
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
        refined_exhibits =[]

        for i, exhibit in enumerate(vlm_exhibits):
            box = np.array(exhibit['bbox'])
            try:
                masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                
                rows = np.any(best_mask, axis=1)
                cols = np.any(best_mask, axis=0)

                if np.any(rows) and np.any(cols):
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    
                    # 【核心修改 1】：使用真实的 Mask 质心代替 BBox 中心
                    y_center, x_center = center_of_mass(best_mask)
                    if np.isnan(y_center) or np.isnan(x_center):
                        center =[int((cmin + cmax) / 2), int((rmin + rmax) / 2)]
                    else:
                        center =[int(x_center), int(y_center)]

                    refined_exhibits.append({
                        'id': f"E{i+1}", 'name': exhibit['name'], 'type': exhibit['type'],
                        'description': exhibit['description'], 'vlm_bbox': exhibit['bbox'],
                        'bbox':[int(cmin), int(rmin), int(cmax), int(rmax)],
                        'center': center, 'area': int(np.sum(best_mask)),
                        'sam_score': float(scores[best_idx]), 'mask': best_mask
                    })
                else:
                    self._add_fallback(exhibit, i, refined_exhibits)
            except Exception as e:
                self._add_fallback(exhibit, i, refined_exhibits)

        return refined_exhibits, image_np

    def _add_fallback(self, exhibit, idx, refined_list):
        bbox = exhibit['bbox']
        refined_list.append({
            'id': f"E{idx+1}", 'name': exhibit['name'], 'type': exhibit['type'],
            'description': exhibit['description'], 'vlm_bbox': bbox,
            'bbox': bbox, 'center': [int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)],
            'area': (bbox[2]-bbox[0])*(bbox[3]-bbox[1]), 'sam_score': 0.8, 'mask': None
        })

    def create_segmentation_visualization(self, image_np, exhibits, output_path):
        height, width = image_np.shape[:2]
        result = np.zeros_like(image_np)
        combined_mask = np.zeros((height, width), dtype=bool)

        for ex in exhibits:
            if ex.get('mask') is not None:
                combined_mask = combined_mask | ex['mask'].astype(bool)
        
        result[combined_mask] = image_np[combined_mask]

        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(result)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        plt.close()


def predict_saliency_heatmap(image_path, exhibits, output_path, sigma=40):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    print("\n" + "="*60 + "\nStep (c): Semantic Saliency Prediction\n" + "="*60)

    # 基础热力图
    saliency = np.zeros((height, width), dtype=np.float32)
    combined_mask = np.zeros((height, width), dtype=bool)

    # 【核心修改 2】：使用精确的点在 Mask 内投射高斯热力
    for ex in exhibits:
        cx, cy = ex['center']
        if ex.get('mask') is not None:
            combined_mask = combined_mask | ex['mask'].astype(bool)
        else:
            x1, y1, x2, y2 = ex['bbox']
            combined_mask[y1:y2, x1:x2] = True
            
        if 0 <= cy < height and 0 <= cx < width:
            saliency[cy, cx] += float(ex.get('sam_score', 1.0)) * 150 # 在质心创建热力峰值

    # 高斯平滑 (产生渐渐发散的热力效果)
    saliency = gaussian_filter(saliency, sigma=sigma)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    # 【核心修改 3】：语义截断！将 Mask 外部的热力值全部清零 (实现图2纯净效果的关键)
    saliency[~combined_mask] = 0.0

    # 映射伪彩色 (使用 JET 或 TURBO 色带，图2通常用这个)
    colormap = plt.get_cmap('jet')
    colored_heatmap = (colormap(saliency)[:, :, :3] * 255).astype(np.uint8)

    # 叠加回原图
    alpha = 0.55 # 调整透明度
    result_array = image.copy()
    
    # 仅在有热力的区域融合原图和热力图
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
    return saliency


def predict_scan_path(image_path, saliency_map, exhibits, output_path, num_fixations=10):
    img_array = np.array(Image.open(image_path).convert('RGB'))
    height, width = img_array.shape[:2]

    # 【生成逻辑保持你原来的基于分数排序的逻辑】
    exhibit_scores =[]
    for ex in exhibits:
        cx, cy = ex['center']
        score = (float(ex.get('sam_score', 0.8)) + (ex['area'] / (width * height))) * 10
        exhibit_scores.append({'exhibit': ex, 'score': score})

    selected = sorted(exhibit_scores, key=lambda x: x['score'], reverse=True)[:num_fixations]
    selected.sort(key=lambda item: item['exhibit']['center'][0]*0.7 + item['exhibit']['center'][1]*0.3)

    fixations = []
    for i, item in enumerate(selected):
        ex = item['exhibit']
        duration = min(40 * (0.6 + item['score']) * (1 + np.log(ex['area'] / 5000 + 1) * 0.3), 250) / 10
        fixations.append({
            'sequence': i + 1, 'exhibit_id': ex['id'], 'exhibit_name': ex['name'],
            'center': ex['center'], 'duration': duration, 'score': item['score']
        })

    # 【核心修改 4】：高对比度、高颜值的轨迹图绘制 (解决图3看不清的问题)
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(img_array)

    if len(fixations) > 1:
        path_x = [f['center'][0] for f in fixations]
        path_y = [f['center'][1] for f in fixations]
        # 画两层线：粗黑底线 + 稍细一点的白线段，形成描边效果
        ax.plot(path_x, path_y, color='black', linewidth=6, alpha=0.9, zorder=2)
        ax.plot(path_x, path_y, color='#F0F0F0', linewidth=3, alpha=1.0, zorder=3)

    for fix in fixations:
        cx, cy = fix['center']
        seq = fix['sequence']
        radius = 22 # 统一节点大小，显得更精美

        # 黑色外圈
        ax.add_patch(Circle((cx, cy), radius+3, facecolor='black', edgecolor='none', alpha=0.9, zorder=4))
        # 亮黄色/白色内圈 (提升学术感)
        ax.add_patch(Circle((cx, cy), radius, facecolor='#FFD700', edgecolor='none', alpha=1.0, zorder=5))
        # 黑色数字
        ax.text(cx, cy, str(seq), color='black', fontsize=18, fontweight='bold', ha='center', va='center', zorder=6)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()
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
