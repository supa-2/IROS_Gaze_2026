# GPU服务器运行说明

## 功能说明

本脚本基于计算机视觉模型预测生成论文图表，**不依赖实际眼动数据**：

| 面板 | 内容 | 方法 |
|------|------|------|
| (a) 原图 | 输入图像 | - |
| (b) 分割掩码 | SAM2 自动分割所有展品、展板 | Meta SAM2 模型 |
| (c) 热力图 | 视觉显著性预测 | 亮度/颜色对比 + 边缘检测 + 中心偏置 |
| (d) 轨迹图 | 扫描路径预测 | 基于显著性和空间布局的预测模型 |

## 环境要求

- Python 3.8+
- CUDA 11.8+ / 12.x
- 显存 >= 4GB
- SAM2 模型

## 安装依赖

```bash
# 克隆 SAM2 仓库 (如果还没有)
git clone https://github.com/facebookresearch/segment-anything-2.git sam2

# 安装 SAM2 及其依赖
pip install -e segment-anything-2

# 或者手动安装所有依赖
pip install git+https://github.com/facebookresearch/segment-anything-2.git
pip install opencv-python matplotlib pillow scipy torch
```

**必需的依赖包**：
```
sam2 (from segment-anything-2)
opencv-python
matplotlib
pillow
scipy
torch
numpy
```

## SAM2 模型

确保 SAM2 模型在以下路径：
```
/home/g/models/iros_agent/models/sam2/sam2_hiera_small.pt
```

如果没有，请下载：
```bash
mkdir -p /home/g/models/iros_agent/models/sam2/
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_small.pt \
     -O /home/g/models/iros_agent/models/sam2/sam2_hiera_small.pt
```

## 项目结构

```
/path/to/iros_agent/
├── sam2/                    # SAM2 代码库
│   ├── configs/
│   └── sam2/
├── scripts/
│   └── generate_paper_figure_gpu.py
├── data/
│   ├── test1.jpg
│   └── test2.jpg
└── run_gpu.sh
```

## 运行方式

### 方式一：一键脚本

```bash
chmod +x run_gpu.sh
./run_gpu.sh
```

### 方式二：单独运行

```bash
# 基本用法
python scripts/generate_paper_figure_gpu.py \
    --image data/test1.jpg \
    --output data/outputs/test1/paper_figure.png

# 自定义参数
python scripts/generate_paper_figure_gpu.py \
    --image /path/to/image.jpg \
    --output /path/to/output.png \
    --sam2-model /path/to/sam2_model.pt \
    --num-fixations 12 \
    --points-per-side 48
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | 必填 | 输入图像路径 |
| `--output` | `data/outputs/paper_figure_predicted.png` | 输出路径 |
| `--sam2-model` | `/home/g/models/.../sam2_hiera_small.pt` | SAM2 模型路径 |
| `--sam2-config` | `sam2/configs/sam2.1/sam2.1_hiera_s.yaml` | SAM2 配置路径 |
| `--num-fixations` | 10 | 预测注视点数量 |
| `--points-per-side` | 32 | SAM2 采样密度 (越大越精细，但越慢) |

## 输出文件

```
data/outputs/
├── test1/
│   ├── paper_figure.png           # 四宫格图表
│   ├── paper_figure_mask.png      # SAM2 分割掩码 (黑色背景)
│   ├── paper_figure_mask_outline.png  # 分割轮廓 (叠加原图)
│   ├── paper_figure_heatmap.png   # 预测热力图
│   └── paper_figure_trajectory.png # 预测轨迹图
└── test2/
    └── ...
```

## 模型说明

### SAM2 分割

使用 `SAM2AutomaticMaskGenerator` 自动检测图像中的所有显著区域：
- 展品（画作、雕塑等）
- 展板、标签
- 其他视觉元素

参数调节：
- `points_per_side`: 采样密度，32-64 适合大多数场景
- `pred_iou_thresh`: 预测置信度阈值，默认 0.88
- `stability_score_thresh`: 稳定性阈值，默认 0.95

### 视觉显著性预测

组合多种视觉特征：
1. **亮度对比** - 高斯差分检测亮度变化
2. **颜色对比** - Lab 空间颜色差异
3. **边缘密度** - Canny 边缘检测
4. **中心偏置** - 模拟人眼中心倾向

### 扫描路径预测

基于显著性和空间布局预测观看顺序：
1. 计算每个分割区域的平均显著性
2. 考虑中心偏置奖励
3. 按空间位置排序（从左到右，从上到下）
4. 预测注视时长（基于显著性和区域大小）

## 故障排查

### ModuleNotFoundError: No module named 'hydra'

```bash
pip install hydra-core
pip install -e segment-anything-2
```

### CUDA out of memory

降低 `points_per_side` 参数：
```bash
python scripts/generate_paper_figure_gpu.py \
    --image data/test1.jpg \
    --points-per-side 16  # 降低采样密度
```

### SAM2 模型未找到

检查模型路径：
```bash
ls -lh /home/g/models/iros_agent/models/sam2/
```
