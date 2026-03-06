#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "IROS Gaze 论文图表生成 (Grounded-SAM 版)"
echo "========================================"

# 创建输出目录
mkdir -p data/outputs/test1
mkdir -p data/outputs/test2

# 处理第一张图
echo "[*] 正在处理第一张图..."
python scripts/generate_paper_figure_gpu.py \
    --image data/test1.jpg \
    --output data/outputs/test1/paper_figure.png \
    --sam2-model models/sam2/sam2_hiera_small.pt \
    --num-fixations 10

# 处理第二张图
echo "[*] 正在处理第二张图..."
python scripts/generate_paper_figure_gpu.py \
    --image data/test2.jpg \
    --output data/outputs/test2/paper_figure.png \
    --sam2-model models/sam2/sam2_hiera_small.pt \
    --num-fixations 10

echo ""
echo "[OK] 全部完成!"
echo "请查看输出目录: data/outputs/test1/ 和 data/outputs/test2/"