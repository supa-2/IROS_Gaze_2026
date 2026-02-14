"""
最终版 VLM + 眭动 → 热图热力图测试
"""

import sys
import os

# 设置 UTF-8 输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# 创建输出目录
os.makedirs("data/outputs", exist_ok=True)

def main():
    print("=" * 70)
    print("VLM + 眭动 → 热图热力图测试")
    print("=" * 70)

    # 测试图片
    print("Creating test image...")
    img = Image.new("RGB", (800, 600), color="#f5f5f5")

    # VLM 识别的区域数据
    # 格式: (x1, y1, x2, y2, label)
    vlM_regions = [
        (100, 150, 300, 350, "青铜器区"),
        (400, 100, 550, 300, "陶瓷展区"),
        (500, 350, 650, 450, "说明牌区")
    ]

    # 眼动数据
    # 格式: (zone_id, duration)
    gaze_data = [
        (100, 150, 300, 30.0),
        (400, 100, 550, 300, 60.0),
        (500, 350, 650, 450, 20.0),
        (600, 400, 800, 550, 15.0)
    ]

    print(f"VLM regions: {len(vlM_regions)}")
    print(f"Gaze data: {len(gaze_data)}")

    # 创建图布
    print("Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(figsize=(14, 10))

    # 左图：原图 + 区域标注
    ax1.imshow(img)
    ax1.set_title("VLM Identified Regions")
    ax1.axis("off")

    # 绘制区域边界框
    colors = ["red", "blue", "green"]
    for i in range(len(vlM_regions)):
        x1, y1, x2, y2 = vlM_regions[i]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1),
        ax1.text(x1, y1-10, vlM_regions[i][3], fontsize=10, color="white")

    # 右图：热力图
    # 计算热力值
    heatmap = np.zeros((600, 800))

    # 为每个区域添加热度（基于观看时长）
    for i, (x1, y1, x2, y2, duration) in zip(vlM_regions, gaze_data):
        for y in range(y1, y1+1):
            for x in range(x1, x1+1):
                heatmap[y, x] = duration

    im = ax2.imshow(heatmap, cmap="jet", alpha=0.7, vmin=0, vmax=100)
    ax2.set_title("Gaze Heatmap (Pixel-level)")

    # 颜色条
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Duration (s)")

    plt.tight_layout()

    # 保存
    output_path = "data/outputs/final_heatmap_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"Heatmap saved to: {output_path}")

    # 显示
    print("Showing heatmap...")
    plt.show()

    print("=" * 70)
    print("Test Complete!")
