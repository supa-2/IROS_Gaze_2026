"""
完整 VLM + 眭动 → 热图热力图 测试

简化版：直接在 Python 中定义所有数据，避免文件编码问题
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在最前面
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import os

# 创建输出目录
os.makedirs('data/outputs', exist_ok=True)

# 创建测试图片
img = Image.new('RGB', (800, 600), color='#f5f5f5')

# 模拟 VLM 识别的区域 (x1, y1, x2, y2)
regions = [
    ('zone_a', [100, 150, 300, 350], '青铜器区'),
    ('zone_b', [400, 100, 550, 300], '陶瓷展区'),
    ('zone_c', [500, 350, 650, 450], '说明牌区')
]

# 模拟眼动数据 (每个区域的观看时长)
gaze_data = [
    ('zone_a', 45.0),
    ('zone_a', 30.0),
    ('zone_b', 60.0),
    ('zone_c', 20.0),
    ('zone_d', 15.0),
]

# 创建图布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# 左图：原图 + 区域标注
ax1.imshow(img)
ax1.set_title("VLM Identified Regions")

# 绘制区域边界框
colors = ['red', 'blue', 'green']
for i, (x1, y1, x2, y2, label) in enumerate(regions):
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor=colors[i], facecolor='none')
    ax1.add_patch(rect)
    ax1.text(x1, y1-10, label, fontsize=10, color='white')

# 右图：热力图
# 计算热力值（基于观看时长）
heatmap = np.zeros((600, 800))

# 为每个区域计算热度
for (zone_id, duration), (x1, y1, x2, y2) in zip(regions, gaze_data):
    # 检查区域是否存在
    if zone_id == 'zone_a':
        x1, y1, x2, y2 = 100, 150, 300, 350
    elif zone_id == 'zone_b':
        x1, y1, x2, y2 = 400, 100, 550, 300
    elif zone_id == 'zone_c':
        x1, y1, x2, y2 = 500, 350, 650, 450
    elif zone_id == 'zone_d':
        x1, y1, x2, y2 = 600, 400, 800, 550

    # 叠加热度值（时长越长越热）
    heatmap[y1:y2, x1:x2] = duration

# 创建热力图
im = ax2.imshow(heatmap, cmap='jet', alpha=0.7, vmin=0, vmax=100)
ax2.set_title("Gaze Heatmap - Pixel Level")

# 添加颜色条
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Heat Intensity (seconds)')

plt.tight_layout()

# 保存
output_path = 'data/outputs/pixel_heatmap_simple.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Heatmap saved to: {output_path}')

# 显示图片
print('Showing heatmap...')
plt.show()

print('Test Complete!')
