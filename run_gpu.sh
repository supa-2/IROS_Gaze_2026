#!/bin/bash
# GPU服务器上运行论文图表生成脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export SAM2_MODEL_PATH=/home/g/models/iros_agent/models/sam2/sam2_hiera_small.pt
export SAM2_CONFIG_PATH=sam2/configs/sam2.1/sam2.1_hiera_s.yaml

# 激活虚拟环境（如果有）
# source /path/to/venv/bin/activate

# 运行脚本
echo "=================================================="
echo "IROS Gaze 论文图表生成 (GPU服务器)"
echo "=================================================="

# 处理 test1.jpg
python scripts/generate_paper_figure_gpu.py \
    --image data/test1.jpg \
    --json data/outputs/pipeline_vlm/FINAL_REPORT_TEST1.json \
    --output data/outputs/paper_figure_test1_sam2.png \
    --sam2-model $SAM2_MODEL_PATH \
    --sam2-config $SAM2_CONFIG_PATH

# 处理 test2.jpg
python scripts/generate_paper_figure_gpu.py \
    --image data/test2.jpg \
    --json data/outputs/pipeline_vlm/FINAL_REPORT_TEST1.json \
    --output data/outputs/paper_figure_test2_sam2.png \
    --sam2-model $SAM2_MODEL_PATH \
    --sam2-config $SAM2_CONFIG_PATH

echo ""
echo "[OK] 全部完成!"
echo "输出文件:"
echo "  - data/outputs/paper_figure_test1_sam2.png"
echo "  - data/outputs/paper_figure_test2_sam2.png"
