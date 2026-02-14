# -*- coding: utf-8 -*-
"""
下载真实博物馆展厅图片并测试眼动热力图
"""

import sys
import os
import io

# UTF-8 output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# 导入热力图可视化器
sys.path.insert(0, os.path.dirname(__file__))
from skills.visualization.pixel_heatmap import GazeHeatmapVisualizer


def download_real_museum_image():
    """下载真实的博物馆展厅图片"""

    # 使用 Unsplash 的高质量博物馆图片
    urls = [
        # 现代艺术博物馆
        "https://images.unsplash.com/photo-1574620038806-7d4c27bf3903?w=1600&h=1000&fit=crop&q=80",
        # 传统博物馆画廊
        "https://images.unsplash.com/photo-1574169208507-8437617818de?w=1600&h=1000&fit=crop&q=80",
        # 艺术展览
        "https://images.unsplash.com/photo-1541367777705-476473dbe62?w=1600&h=1000&fit=crop&q=80",
        # 博物馆展品
        "https://images.unsplash.com/photo-1566127444979-b3d2b654e3d8?w=1600&h=1000&fit=crop&q=80",
    ]

    output_path = "data/real_museum.jpg"
    os.makedirs("data", exist_ok=True)

    for i, url in enumerate(urls):
        try:
            print(f"[Download] Attempting URL {i+1}/{len(urls)}...")
            print(f"  URL: {url[:60]}...")

            # 设置请求头
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')

            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()

                with open(output_path, 'wb') as f:
                    f.write(data)

                # 验证图片
                img = Image.open(output_path)
                print(f"[Download] Success! Image size: {img.size[0]} x {img.size[1]}")
                print(f"[Download] Saved to: {output_path}")

                return output_path

        except Exception as e:
            print(f"[Download] Failed: {e}")
            continue

    # 如果所有URL都失败，创建模拟真实场景的图片
    print("[Fallback] Creating realistic museum scene...")
    return create_realistic_museum_scene()


def create_realistic_museum_scene():
    """创建逼真的博物馆场景（使用渐变和纹理）"""
    output_path = "data/realistic_museum.jpg"

    # 创建画布 - 博物馆米色墙面
    width, height = 1600, 1000
    img = Image.new("RGB", (width, height), color="#F5F5DC")
    draw = ImageDraw.Draw(img)

    # 背景：墙面渐变效果（用矩形模拟）
    for i in range(height):
        gray_val = int(245 - i * 0.02)
        color = f"#{gray_val:02x}{gray_val:02x}{gray_val-10:02x}"
        draw.rectangle([(0, i), (width, i+1)], fill=color)

    # 地板
    draw.rectangle([0, 700, width, height], fill="#8B7355")

    # 左侧展柜 - 青铜器展示
    cabinet_x1, cabinet_y1 = 100, 150
    cabinet_x2, cabinet_y2 = 450, 650

    # 展柜外框
    draw.rectangle([cabinet_x1, cabinet_y1, cabinet_x2, cabinet_y2],
                 fill="#4A3728", outline="#2F1F1F", width=5)

    # 展柜内部玻璃效果
    draw.rectangle([cabinet_x1+15, cabinet_y1+15, cabinet_x2-15, cabinet_y2-100],
                 fill="#E6E6FA", outline="#B0B0FF", width=2)

    # 青铜展品（深色圆形）
    draw.ellipse([200, 250, 350, 400], fill="#8B4513", outline="#CD7F32", width=8)

    # 中间展台 - 陶瓷展品
    pedestal_x1, pedestal_y1 = 550, 300
    pedestal_x2, pedestal_y2 = 900, 650

    # 展台底座
    draw.rectangle([pedestal_x1, pedestal_y1, pedestal_x2, pedestal_y2],
                 fill="#DEB887", outline="#8B7355", width=5)

    # 陶瓷展品（浅色花瓶形状）
    draw.ellipse([650, 320, 800, 450], fill="#87CEEB", outline="#4682B4", width=6)
    draw.rectangle([680, 280, 770, 330], fill="#B0E0E6", outline="#4682B4", width=3)

    # 右侧展墙 - 书画
    wall_x1, wall_y1 = 1000, 100
    wall_x2, wall_y2 = 1450, 650

    # 展墙背景
    draw.rectangle([wall_x1, wall_y1, wall_x2, wall_y2],
                 fill="#FFFAF0", outline="#DAA520", width=5)

    # 书画框
    draw.rectangle([1100, 150, 1400, 400], fill="#FFF8DC", outline="#BDB76B", width=8)
    draw.rectangle([1120, 170, 1380, 380], fill="#FFEFD5", outline="#EEE8AA", width=3)

    # 说明牌（底部）
    sign_x1, sign_y1 = 400, 750
    sign_x2, sign_y2 = 1100, 850

    draw.rectangle([sign_x1, sign_y1, sign_x2, sign_y2],
                 fill="#2F4F4F", outline="#1C3333", width=4)

    # 添加一些"文字"效果
    for x in range(450, 1050, 100):
        draw.rectangle([x, 780, x+60, 790], fill="#696969", outline="")

    # 保存
    img.save(output_path, quality=95)
    print(f"[Created] Realistic museum scene: {output_path}")

    return output_path


def main():
    print("=" * 70)
    print("  真实博物馆场景 + 眼动热力图测试")
    print("=" * 70)

    # 1. 下载/创建真实博物馆图片
    print("\n[Step 1] Loading real museum image...")
    image_path = download_real_museum_image()

    # 2. 初始化可视化器
    print("\n[Step 2] Initializing heatmap visualizer...")
    viz = GazeHeatmapVisualizer(image_path, sigma=80)
    print(f"    Image size: {viz.image_width} x {viz.image_height}")

    # 3. 添加 VLM 识别的区域（模拟识别结果）
    print("\n[Step 3] Adding VLM detected regions...")
    vlm_regions = [
        {
            "id": "bronze_exhibit",
            "label": "Bronze Ding Exhibit",
            "type": "Exhibit",
            "bbox": [100, 150, 450, 650]
        },
        {
            "id": "pottery_pedestal",
            "label": "Pottery Vase Display",
            "type": "Exhibit",
            "bbox": [550, 300, 900, 650]
        },
        {
            "id": "painting_wall",
            "label": "Art Painting Wall",
            "type": "Exhibit",
            "bbox": [1000, 100, 1450, 650]
        },
        {
            "id": "info_sign",
            "label": "Information Sign",
            "type": "Label",
            "bbox": [400, 750, 1100, 850]
        },
    ]
    viz.add_vlm_regions(vlm_regions)
    print(f"    Added {len(vlm_regions)} regions")

    # 4. 模拟真实眼动数据 - 更分散的凝视点
    print("\n[Step 4] Adding realistic gaze fixation points...")

    # 青铜展品 - 观众停留最久（主要展品）
    viz.add_gaze_record("bronze_exhibit", 35.0, x=275, y=325)
    viz.add_gaze_record("bronze_exhibit", 28.0, x=250, y=350)
    viz.add_gaze_record("bronze_exhibit", 25.0, x=300, y=380)
    viz.add_gaze_record("bronze_exhibit", 22.0, x=275, y=400)
    viz.add_gaze_record("bronze_exhibit", 18.0, x=260, y=360)

    # 陶瓷展台 - 中等关注
    viz.add_gaze_record("pottery_pedestal", 30.0, x=725, y=380)
    viz.add_gaze_record("pottery_pedestal", 25.0, x=750, y=400)
    viz.add_gaze_record("pottery_pedestal", 20.0, x=700, y=420)
    viz.add_gaze_record("pottery_pedestal", 15.0, x=730, y=360)

    # 书画墙 - 快速浏览
    viz.add_gaze_record("painting_wall", 18.0, x=1250, y=275)
    viz.add_gaze_record("painting_wall", 15.0, x=1280, y=300)
    viz.add_gaze_record("painting_wall", 12.0, x=1220, y=320)
    viz.add_gaze_record("painting_wall", 10.0, x=1260, y=280)

    # 说明牌 - 简单扫视
    viz.add_gaze_record("info_sign", 8.0, x=750, y=800)
    viz.add_gaze_record("info_sign", 6.0, x=800, y=820)
    viz.add_gaze_record("info_sign", 5.0, x=700, y=810)

    print(f"    Total fixation points: {len(viz.fixations)}")

    # 5. 获取统计信息
    print("\n[Step 5] Region statistics:")
    stats = viz.get_region_statistics()
    for region_id, stat in stats.items():
        print(f"    {stat['label']}:")
        print(f"      Fixations: {stat['fixation_count']}")
        print(f"      Total duration: {stat['total_duration']:.1f}s")
        print(f"      Avg duration: {stat['avg_duration']:.1f}s")

    # 6. 生成并排对比图
    print("\n[Step 6] Generating side-by-side comparison...")
    os.makedirs("data/outputs", exist_ok=True)
    output_compare = viz.visualize_side_by_side(
        output_path="data/outputs/real_museum_comparison.png",
        show=False
    )

    # 7. 生成单独叠加图（高透明度）
    print("\n[Step 7] Generating overlay visualization...")
    output_overlay = viz.visualize_overlay(
        output_path="data/outputs/real_museum_overlay.png",
        show=False,
        alpha=0.55,
        cmap='jet'
    )

    # 8. 生成高清叠加图
    print("\n[Step 8] Generating high-res overlay...")
    output_hires = viz.visualize_overlay(
        output_path="data/outputs/real_museum_overlay_hires.png",
        show=False,
        alpha=0.5,
        cmap='hot'  # 使用 hot 色彩映射（更红）
    )

    print("\n" + "=" * 70)
    print("  Test Complete!")
    print(f"  Original image: {image_path}")
    print(f"  Comparison: {output_compare}")
    print(f"  Overlay (jet): {output_overlay}")
    print(f"  Overlay (hot): {output_hires}")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
