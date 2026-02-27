#!/bin/bash
# 在模型服务器上运行消融实验的脚本
#
# 使用方法：
# 1. 将整个项目复制到模型服务器
# 2. 在模型服务器上运行此脚本

set -e

echo "========================================="
echo "  IROS 2026 - 消融实验 (使用微调模型)"
echo "========================================="
echo "模型路径: /home/g/models/qwen2.5-32b-int4"
echo ""

# 检查 vLLM 是否安装
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "[!] vLLM 未安装，正在安装..."
    pip install vllm
fi

# 检查模型是否存在
if [ ! -d "/home/g/models/qwen2.5-32b-int4" ]; then
    echo "[!] 错误: 模型目录不存在: /home/g/models/qwen2.5-32b-int4"
    exit 1
fi

echo "[+] 模型目录找到"
echo ""

# 运行消融实验
echo "[*] 开始运行消融实验..."
echo "----------------------------------------"

python3 scripts/run_vllm_ablation.py \
    --model /home/g/models/qwen2.5-32b-int4 \
    --map TH \
    --data data/processed/test_sequences.json

echo ""
echo "========================================="
echo "  实验完成！"
echo "========================================="
echo ""
echo "结果保存在: data/outputs/vllm_ablation/"
echo ""
echo "查看结果:"
echo "  cat data/outputs/vllm_ablation/ablation_results.json"
echo ""
