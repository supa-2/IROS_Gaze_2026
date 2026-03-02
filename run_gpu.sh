#!/bin/bash
# GPU服务器上运行论文图表生成脚本 (模型预测版本)

# 设置环境变量（使用相对路径）
export CUDA_VISIBLE_DEVICES=0
export SAM2_MODEL_PATH=models/sam2/sam2_hiera_small.pt
export SAM2_CONFIG_NAME=sam2.1_hiera_s  # 配置名称，不是文件路径

echo "=================================================="
echo "IROS Gaze 论文图表生成 (GPU服务器 - 模型预测版本)"
echo "=================================================="
echo ""
echo "功能说明:"
echo "  (a) 原图"
echo "  (b) SAM2 自动分割掩码 (分割所有展品和展板)"
echo "  (c) 视觉显著性预测热力图"
echo "  (d) 扫描路径预测轨迹图"
echo ""
echo "=================================================="

# 处理 test1.jpg
echo ""
echo "[*] 处理 test1.jpg..."
mkdir -p data/outputs/test1
python scripts/generate_paper_figure_gpu.py \
    --image data/test1.jpg \
    --output data/outputs/test1/paper_figure.png \
    --sam2-model $SAM2_MODEL_PATH \
    --sam2-config $SAM2_CONFIG_NAME \
    --num-fixations 10 \
    --points-per-side 32

# 处理 test2.jpg
echo ""
echo "[*] 处理 test2.jpg..."
mkdir -p data/outputs/test2
python scripts/generate_paper_figure_gpu.py \
    --image data/test2.jpg \
    --output data/outputs/test2/paper_figure.png \
    --sam2-model $SAM2_MODEL_PATH \
    --sam2-config $SAM2_CONFIG_NAME \
    --num-fixations 10 \
    --points-per-side 32

echo ""
echo "[OK] 全部完成!"
echo ""
echo "输出文件:"
echo "  - data/outputs/test1/"
echo "    ├── paper_figure.png           (四宫格图表)"
echo "    ├── paper_figure_mask.png      (分割掩码)"
echo "    ├── paper_figure_mask_outline.png (分割轮廓)"
echo "    ├── paper_figure_heatmap.png   (预测热力图)"
echo "    └── paper_figure_trajectory.png (预测轨迹)"
echo ""
echo "  - data/outputs/test2/"
echo "    └── (同上)"
