# GPU服务器运行说明

## 环境要求

- Python 3.8+
- CUDA 11.8+ / 12.x
- 显存 >= 4GB

## 安装依赖

```bash
# 安装 SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# 安装其他依赖
pip install numpy matplotlib pillow scipy torch
```

## SAM2 模型

确保 SAM2 模型在以下路径：
```
/home/g/models/iros_agent/models/sam2/sam2_hiera_small.pt
```

如果没有，请下载：
```bash
# 创建目录
mkdir -p /home/g/models/iros_agent/models/sam2/

# 下载模型
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_small.pt -O /home/g/models/iros_agent/models/sam2/sam2_hiera_small.pt
```

## 运行方式

### 方式一：使用一键脚本

```bash
chmod +x run_gpu.sh
./run_gpu.sh
```

### 方式二：单独运行

```bash
# 生成 test1 的图表
python scripts/generate_paper_figure_gpu.py \
    --image data/test1.jpg \
    --json data/outputs/pipeline_vlm/FINAL_REPORT_TEST1.json \
    --output data/outputs/paper_figure_test1_sam2.png

# 生成 test2 的图表
python scripts/generate_paper_figure_gpu.py \
    --image data/test2.jpg \
    --json data/outputs/pipeline_vlm/FINAL_REPORT_TEST1.json \
    --output data/outputs/paper_figure_test2_sam2.png
```

### 自定义参数

```bash
python scripts/generate_paper_figure_gpu.py \
    --image /path/to/image.jpg \
    --json /path/to/gaze_data.json \
    --output /path/to/output.png \
    --sam2-model /path/to/sam2_model.pt \
    --sam2-config sam2/configs/sam2.1/sam2.1_hiera_s.yaml
```

## 输出

脚本会生成：
- 四宫格图表（原尺寸 3000x2000）
- 单独的分割掩码图
- 单独的热力图
- 单独的轨迹图
