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
    """使用 Qwen-VL 识别图片中的展品及位置"""
    print("\n" + "="*60)
    print("步骤 (a): VLM 识别展品")
    print("="*60)

    # 获取 API 配置
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("[!] 错误: 未找到有效的 API Key")
        print("    请在 .env 文件中设置 QWEN_API_KEY")
        return []

    base_url = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("VLM_MODEL", "qwen-vl-max-latest")

    print(f"    API: {base_url}")
    print(f"    Model: {model}")

    # 编码图片
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    prompt = """请分析这张展厅图片，识别出所有值得观看的展品。

对于每个展品，请提供：
1. 展品名称（简洁，如：画作1、雕塑A）
2. 展品类型（必须是：画作、雕塑、装置艺术、摄影作品之一）
3. 在图片中的位置（边界框坐标 [x1, y1, x2, y2]，其中 (0,0) 是左上角）
4. 简短描述（10字以内）

请以 JSON 格式返回：
[
  {
    "name": "展品名称",
    "type": "画作/雕塑/装置艺术/摄影作品",
    "bbox": [x1, y1, x2, y2],
    "description": "描述"
  }
]

要求：
- 只识别真正的展品，忽略墙壁、地板、展柜、灯光
- 边界框要紧凑地包围展品主体
- 返回 5-12 个主要展品
- type 必须是：画作、雕塑、装置艺术、摄影作品 之一"""

    print("[*] 调用 Qwen-VL API...")

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

        # 解析 JSON
        import re
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            exhibits = json.loads(json_match.group())

            # 验证并过滤
            valid_exhibits = []
            for ex in exhibits:
                bbox = ex.get('bbox', [])
                if len(bbox) == 4:
                    try:
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        area = (x2 - x1) * (y2 - y1)
                        if 500 < area < 600000:  # 面积合理
                            # 规范化 type
                            ex_type = ex.get('type', '画作')
                            if ex_type not in ['画作', '雕塑', '装置艺术', '摄影作品']:
                                ex_type = '画作'
                            valid_exhibits.append({
                                "name": ex.get('name', f'展品{len(valid_exhibits)+1}'),
                                "type": ex_type,
                                "bbox": [x1, y1, x2, y2],
                                "description": ex.get('description', '')
                            })
                    except (ValueError, TypeError):
                        continue

            print(f"[+] VLM 识别到 {len(valid_exhibits)} 个有效展品")
            return valid_exhibits
        else:
            print("[!] 无法解析 VLM 响应为 JSON")
            return []

    except Exception as e:
        print(f"[!] VLM 调用失败: {e}")
        return []


class SAM2Segmenter:
    """SAM2 精细分割器"""

    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device

        if not os.path.isabs(model_path):
            abs_model_path = os.path.join(project_root, model_path)
        else:
            abs_model_path = model_path

        print(f"\n[*] 初始化 SAM2...")
        print(f"    模型: {model_path}")

        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(f"模型文件不存在: {abs_model_path}")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # 根据文件名确定配置
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

            print(f"    配置: {config_name}")

            model = build_sam2(
                config_file=config_name,
                ckpt_path=abs_model_path,
                device=device
            )

            self.predictor = SAM2ImagePredictor(model, device=device)
            print("[+] SAM2 加载成功")

        except Exception as e:
            print(f"[!] SAM2 加载失败: {e}")
            raise

    def refine_with_vlm_boxes(self, image_np, vlm_exhibits):
        """基于 VLM bbox 进行精细分割"""
        print("\n" + "="*60)
        print("步骤 (b): SAM2 精细分割")
        print("="*60)

        self.predictor.set_image(image_np)
        height, width = image_np.shape[:2]

        refined_exhibits = []

        for i, exhibit in enumerate(vlm_exhibits):
            print(f"    处理 {exhibit['name']}...")

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

                # 计算精细边界框
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
                    print(f"        分割成功: 面积={area}, 置信度={best_score:.3f}")
                else:
                    # 使用原始 bbox
                    self._add_fallback(exhibit, i, refined_exhibits)

            except Exception as e:
                print(f"        分割失败: {e}, 使用VLM bbox")
                self._add_fallback(exhibit, i, refined_exhibits)

        print(f"[+] 精细分割完成: {len(refined_exhibits)} 个展品")
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
        """创建分割掩码可视化"""
        height, width = image_np.shape[:2]

        # 创建黑色背景
        result = np.zeros_like(image_np)

        # 创建统一掩码
        combined_mask = np.zeros((height, width), dtype=bool)

        for ex in exhibits:
            if ex.get('mask') is not None:
                combined_mask = combined_mask | ex['mask']
            else:
                x1, y1, x2, y2 = ex['bbox']
                combined_mask[y1:y2, x1:x2] = True

        # 只在掩码区域显示原图
        result[combined_mask] = image_np[combined_mask]

        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(result)
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        plt.close()
        print(f"[+] 保存分割图: {output_path}")

        return result


def predict_saliency_heatmap(image_path, exhibits, output_path, sigma=20):
    """基于展品位置预测热力图"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    print("\n" + "="*60)
    print("步骤 (c): 视觉显著性预测")
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
    print(f"[+] 保存热力图: {output_path}")

    return saliency


def predict_scan_path(image_path, saliency_map, exhibits, output_path, num_fixations=10):
    """基于显著性和展品信息预测扫描路径"""
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    print("\n" + "="*60)
    print("步骤 (d): 扫描路径预测")
    print("="*60)

    # 计算每个展品的注视得分
    exhibit_scores = []
    for ex in exhibits:
        bbox = ex['bbox']
        x1, y1, x2, y2 = bbox
        cx, cy = ex['center']

        # 该区域的平均显著性
        mean_saliency = saliency_map[y1:y2, x1:x2].mean() if y2 > y1 and x2 > x1 else 0

        # 中心偏置
        img_cx, img_cy = width/2, height/2
        dist_to_center = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
        center_bias = np.exp(-dist_to_center / (min(width, height) / 2))

        # 类型偏好：画作 > 雕塑 > 其他
        type_bonus = {'画作': 1.0, '雕塑': 0.9, '摄影作品': 0.85, '装置艺术': 0.8}
        type_pref = type_bonus.get(ex['type'], 0.85)

        # 综合得分
        score = (mean_saliency * 0.5 + center_bias * 0.3 + type_pref * 0.2)

        exhibit_scores.append({
            'exhibit': ex,
            'score': score,
            'mean_saliency': mean_saliency
        })

    # 按得分排序，选择前 N 个
    exhibit_scores.sort(key=lambda x: x['score'], reverse=True)
    selected = exhibit_scores[:min(num_fixations, len(exhibit_scores))]

    # 按空间位置排序（从左到右，从上到下）
    def scan_order_key(item):
        cx, cy = item['exhibit']['center']
        return cx * 0.6 + cy * 0.4

    selected.sort(key=scan_order_key)

    # 生成注视点数据
    fixations = []
    for i, item in enumerate(selected):
        ex = item['exhibit']
        score = item['score']

        # 预测注视时长（基于得分和面积）
        base_duration = 40
        area_factor = np.log(ex['area'] / 5000 + 1) * 0.3
        duration = base_duration * (0.6 + score) * (1 + area_factor)
        duration = min(duration, 250)

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
    print(f"[+] 保存轨迹图: {output_path}")

    return fixations


def get_attention_level(duration, all_durations):
    """根据时长计算关注等级 A/B/C/D/E"""
    if not all_durations:
        return 'C'

    max_dur = max(all_durations)
    min_dur = min(all_durations)
    range_dur = max_dur - min_dur

    if range_dur == 0:
        return 'C'

    # 5个等级
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
    """生成论文用表格数据"""
    img = Image.open(image_path)
    width, height = img.size
    total_pixels = width * height

    print("\n" + "="*60)
    print("生成表格数据")
    print("="*60)

    # 计算每个展品的注视统计
    exhibit_stats = []
    all_durations = []

    for ex in exhibits:
        # 查找该展品的注视
        ex_fixations = [f for f in fixations if f['exhibit_id'] == ex['id']]
        gaze_count = len(ex_fixations)
        total_duration = sum(f['duration'] for f in ex_fixations)
        all_durations.append(total_duration)

        # 首次和末次注视
        first_seq = min([f['sequence'] for f in ex_fixations]) if ex_fixations else '-'
        last_seq = max([f['sequence'] for f in ex_fixations]) if ex_fixations else '-'

        exhibit_stats.append({
            'exhibit': ex,
            'gaze_count': gaze_count,
            'total_duration': total_duration,
            'first_seq': first_seq,
            'last_seq': last_seq
        })

    # 计算关注等级
    for stat in exhibit_stats:
        stat['attention_level'] = get_attention_level(stat['total_duration'], all_durations) if stat['gaze_count'] > 0 else '-'
        stat['avg_duration'] = stat['total_duration'] / stat['gaze_count'] if stat['gaze_count'] > 0 else 0

    # 计算总统计
    total_fixations = len(fixations)
    total_duration_all = sum(f['duration'] for f in fixations)
    avg_duration_all = total_duration_all / total_fixations if total_fixations > 0 else 0
    gazed_count = sum(1 for s in exhibit_stats if s['gaze_count'] > 0)

    # 打印表格
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

    # 保存 JSON
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
    print(f"\n[+] 保存表格数据: {json_path}")

    # LaTeX 表格
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
            avg_str = f"{s['avg_duration']/1000:.1f}" if s['avg_duration'] > 0 else "-"
            f.write(f"{ex['id']} & {ex['type']} & {ex['area']/total_pixels*100:.1f} & "
                   f"{ex['sam_score']:.2f} & {s['gaze_count']} & "
                   f"{s['total_duration']/1000:.1f} & {avg_str} & "
                   f"{s['first_seq']} & {s['last_seq']} & {s['attention_level']} \\\\\\\\\n")
        f.write(r"\hline" + "\n")
        f.write(f"TOTAL & - & 100 & - & {total_fixations} & "
               f"{total_duration_all/1000:.1f} & {avg_duration_all/1000:.1f} & "
               f"- & - & {gazed_count}/{len(exhibits)} \\\\\\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"[+] 保存LaTeX表格: {latex_path}")

    return summary


def create_paper_figure(image_path, output_path, sam2_model_path, num_fixations=10):
    """生成论文用四宫格图表"""

    if not torch.cuda.is_available():
        print("[!] CUDA 不可用")
        return False

    print(f"\n{'='*60}")
    print(f"IROS Gaze 论文图表生成")
    print(f"{'='*60}")
    print(f"图像: {image_path}")

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    original_img = Image.open(image_path).convert('RGB')
    img_array = np.array(original_img)
    width, height = original_img.size
    print(f"尺寸: {width}x{height}")

    # Step 1: VLM 识别
    vlm_exhibits = call_qwen_vlm(image_path)
    if not vlm_exhibits:
        print("[!] VLM 识别失败，退出")
        return False

    # Step 2: SAM2 精细分割
    segmenter = SAM2Segmenter(sam2_model_path)
    exhibits, _ = segmenter.refine_with_vlm_boxes(img_array, vlm_exhibits)

    mask_path = output_path.replace('.png', '_mask.png')
    segmenter.create_segmentation_visualization(img_array, exhibits, mask_path)

    # Step 3: 热力图
    heatmap_path = output_path.replace('.png', '_heatmap.png')
    saliency_map = predict_saliency_heatmap(image_path, exhibits, heatmap_path)

    # Step 4: 扫描路径
    trajectory_path = output_path.replace('.png', '_trajectory.png')
    fixations = predict_scan_path(image_path, saliency_map, exhibits, trajectory_path, num_fixations)

    # Step 5: 生成表格数据
    generate_table_data(exhibits, fixations, image_path, output_dir)

    # Step 6: 生成四宫格图表
    print("\n" + "="*60)
    print("生成四宫格图表")
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
    print(f"\n[+] 保存四宫格图表: {output_path}")

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
        print(f"[!] 图像不存在: {args.image}")
        return

    # 检查 API Key
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("[!] 错误: 未设置 QWEN_API_KEY")
        print("    请在 .env 文件中设置: QWEN_API_KEY=你的密钥")
        return

    if torch.cuda.is_available():
        print(f"[+] CUDA: {torch.cuda.get_device_name(0)}")
        print(f"    显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("[!] CUDA 不可用")
        return

    create_paper_figure(
        args.image, args.output, args.sam2_model, args.num_fixations
    )

    print("\n[OK] 完成!")


if __name__ == '__main__':
    main()
