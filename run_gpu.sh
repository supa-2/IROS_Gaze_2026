#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "IROS Gaze 论文图表生成"
echo "========================================"

mkdir -p data/outputs/test1
python scripts/generate_paper_figure_gpu.py \
    --image data/test1.jpg \
    --output data/outputs/test1/paper_figure.png \
    --sam2-model models/sam2/sam2_hiera_small.pt

mkdir -p data/outputs/test2
python scripts/generate_paper_figure_gpu.py \
    --image data/test2.jpg \
    --output data/outputs/test2/paper_figure.png \
    --sam2-model models/sam2/sam2_hiera_small.pt

echo ""
echo "[OK] 完成!"
echo "输出: data/outputs/test1/ data/outputs/test2/"
