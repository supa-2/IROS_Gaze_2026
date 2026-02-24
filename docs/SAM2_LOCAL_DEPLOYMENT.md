# SAM 2 本地部署指南

## 概述

SAM 2 (Segment Anything Model 2) 已成功部署到本地，无需调用 API，所有计算在本地完成。

## 模型信息

当前安装的模型：
- **模型**: SAM 2.1 Hiera Small
- **大小**: 176 MB
- **路径**: `sam2/checkpoints/sam2.1_hiera_small.pt`
- **设备**: CPU（可配置为 CUDA）

## 可用模型

| 模型 | 大小 | 下载链接 |
|------|------|----------|
| tiny | 38.9 MB | [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) |
| small | 46.0 MB | [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) |
| base_plus | 80.8 MB | [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) |
| large | 224.4 MB | [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |

## 使用方式

### 1. 测试脚本

```bash
# 使用默认图像测试
python scripts/test_sam2_local.py

# 指定图像和模型
python scripts/test_sam2_local.py --image data/R.jpg --model small

# 调整最小区域面积
python scripts/test_sam2_local.py --image data/R.jpg --min-area 3000
```

### 2. Python 代码

```python
from skills.segmentation.sam2_local import SAM2LocalSegmenter

# 初始化分割器
segmenter = SAM2LocalSegmenter(model_size='small', device='cpu')

# 分割图像
exhibits = segmenter.segment_exhibits(
    image_path='data/R.jpg',
    min_area=5000,
    max_area=500000
)

# exhibits 是一个列表，每个元素包含：
# - id: 展品ID (EX-000, EX-001, ...)
# - bbox: 边界框 [x1, y1, x2, y2]
# - area: 区域面积
# - score: 预测置信度
# - mask: 分割掩码
```

### 3. 点预测

```python
from skills.segmentation.sam2_local import SAM2LocalSegmenter
from PIL import Image
import numpy as np

# 初始化
segmenter = SAM2LocalSegmenter(model_size='small', device='cpu')

# 加载图像
image = Image.open('data/R.jpg').convert('RGB')
image_array = np.array(image)

# 设置图像
segmenter.set_image(image_array)

# 在指定点预测掩码
masks, scores, logits = segmenter.predict(
    point_coords=np.array([[550, 300]]),  # 图像中心点
    point_labels=np.array([1])  # 1=前景, 0=背景
)

# masks 是一个 (N, H, W) 数组
print(f"生成了 {len(masks)} 个掩码")
print(f"最佳掩码分数: {scores[0]:.3f}")
```

### 4. 自动分割

```python
# 自动生成所有掩码
masks = segmenter.auto_segment(
    image_array,
    points_per_side=32,  # 每边的点数（越多越精细，但越慢）
    pred_iou_thresh=0.9,  # 预测 IoU 阈值
    stability_score_thresh=0.96,  # 稳定性分数阈值
    min_mask_region_area=100,  # 最小掩码区域面积
)

print(f"找到了 {len(masks)} 个区域")
```

## 输出示例

分割结果会保存到 `data/outputs/sam2_local/` 目录：

```
data/outputs/sam2_local/
├── segmentation_result.png  # 可视化结果
└── exhibits.json           # JSON 格式的展品数据
```

JSON 格式示例：
```json
{
  "image_path": "data/R.jpg",
  "model": "small",
  "num_exhibits": 8,
  "exhibits": [
    {
      "id": "EX-000",
      "bbox": [923, 477, 1099, 599],
      "area": 20470,
      "score": 0.988
    },
    ...
  ]
}
```

## 性能优化

### CPU vs GPU

```python
# 使用 CPU（默认）
segmenter = SAM2LocalSegmenter(model_size='small', device='cpu')

# 使用 GPU（如果可用）
segmenter = SAM2LocalSegmenter(model_size='small', device='cuda')
```

### 模型选择

- **tiny**: 最快，适合实时应用
- **small**: 平衡性能和速度（推荐）
- **base_plus**: 更高精度
- **large**: 最高精度，但最慢

### 参数调整

```python
# 加快速度（降低精度）
masks = segmenter.auto_segment(
    image_array,
    points_per_side=16,  # 减少点数
    pred_iou_thresh=0.85,  # 降低阈值
    stability_score_thresh=0.90
)

# 提高精度（变慢）
masks = segmenter.auto_segment(
    image_array,
    points_per_side=64,  # 增加点数
    pred_iou_thresh=0.95,  # 提高阈值
    stability_score_thresh=0.98
)
```

## 注意事项

1. **CUDA 扩展警告**: 如果看到 `cannot import name '_C'` 警告，可以忽略。这只是表示 CUDA 后处理扩展没有编译，不影响基本功能。

2. **内存使用**: large 模型在 CPU 上可能消耗较多内存。如果遇到内存问题，使用 small 或 tiny 模型。

3. **图像大小**: SAM 2 对图像大小没有严格限制，但非常大的图像可能会很慢。建议先将图像调整到合理大小（如 1920x1080 以下）。

## 故障排除

### 模型文件不存在

```
FileNotFoundError: Model checkpoint not found: sam2/checkpoints/sam2.1_hiera_small.pt
```

**解决方案**: 下载模型文件到 `sam2/checkpoints/` 目录

```bash
wget -O sam2/checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### CUDA 相关错误

```
RuntimeError: Found no NVIDIA driver on your system.
```

**解决方案**: 使用 CPU 设备

```python
segmenter = SAM2LocalSegmenter(model_size='small', device='cpu')
```

### 导入错误

```
ModuleNotFoundError: No module named 'sam2'
```

**解决方案**: 安装 SAM 2

```bash
uv pip install -e ./sam2
```

## 与 VLM 集成

SAM 2 可以与 VLM 结合使用：

```python
# 1. 使用 SAM 2 分割图像
from skills.segmentation.sam2_local import SAM2LocalSegmenter
segmenter = SAM2LocalSegmenter(model_size='small', device='cpu')
exhibits = segmenter.segment_exhibits('data/R.jpg')

# 2. 使用 VLM 识别展品内容
from scripts.unified_visualization import VLMRecognizer
vlm = VLMRecognizer()
vlm_result = vlm.recognize('data/R.jpg')

# 3. 结合两种结果
for sam_exhibit in exhibits:
    for vlm_exhibit in vlm_result:
        # 计算 IoU 或中心点距离来匹配
        ...
```
