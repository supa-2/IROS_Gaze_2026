#!/bin/bash
# GPU服务器上运行论文图表生成脚本 (模型预测版本)

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export SAM2_MODEL_PATH=/home/g/models/iros_agent/models/sam2/sam2_hiera_small.pt
export SAM2_CONFIG_PATH=sam2/configs/sam2.1/sam2.1_hiera_s.yaml

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
python scripts/generate_paper_figure_gpu.py \
    --image data/test1.jpg \
    --output data/outputs/paper_figure_test1_predicted.png \
    --sam2-model $SAM2_MODEL_PATH \
    --sam2-config $SAM2_CONFIG_PATH \
    --num-fixations 10 \
    --points-per-side 32

# 处理 test2.jpg
echo ""
echo "[*] 处理 test2.jpg..."
python scripts/generate_paper_figure_gpu.py \
    --image data/test2.jpg \
    --output data/outputs/paper_figure_test2_predicted.png \
    --sam2-model $SAM2_MODEL_PATH \
    --sam2-config $SAM2_CONFIG_PATH \
    --num-fixations 10 \
    --points-per-side 32

echo ""
echo "[OK] 全部完成!"
echo ""
echo "输出文件:"
echo "  - data/outputs/paper_figure_test1_predicted.png"
echo "  - data/outputs/paper_figure_test2_predicted.png"
echo ""
echo "单独的子图:"
echo "  - *_mask.png       (SAM2 分割掩码)"
echo "  - *_mask_outline.png (分割轮廓)"
echo "  - *_heatmap.png    (预测热力图)"
echo "  - *_trajectory.png (预测轨迹)"
