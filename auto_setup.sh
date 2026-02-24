#!/bin/bash
# Eye-LLM 全自动环境配置脚本
# 无需任何人工干预，自动处理所有错误并重试

set -e  # 遇到错误退出，但在关键步骤会处理

PROJECT_DIR="/home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  Eye-LLM 自动环境配置"
echo "  开始时间: $(date)"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 重试函数
retry() {
    local max_attempts=$1
    shift
    local cmd="$@"
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        log_info "尝试 $attempt/$max_attempts: $cmd"
        if eval "$cmd"; then
            log_info "成功！"
            return 0
        else
            log_warn "失败，等待 3 秒后重试..."
            sleep 3
            ((attempt++))
        fi
    done

    log_error "达到最大重试次数，跳过此步骤"
    return 1
}

# 1. 检查 Python 版本
log_info "====== 步骤 1: 检查 Python 版本 ======"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log_info "Python 版本: $PYTHON_VERSION"

    # 检查是否 >= 3.11
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
        log_error "Python 版本过低，需要 3.11+"
        log_info "尝试使用 pyenv 安装 Python 3.11..."

        # 检查 pyenv
        if ! command -v pyenv &> /dev/null; then
            log_info "安装 pyenv..."
            curl -fsSL https://pyenv.run | bash
            export PATH="$HOME/.pyenv/bin:$PATH"
        fi

        log_info "通过 pyenv 安装 Python 3.11..."
        pyenv install 3.11.0 -s || pyenv install 3.12.0 -s
        pyenv local 3.11.0 || pyenv local 3.12.0
    fi
else
    log_error "未找到 Python，请先安装 Python 3.11+"
    exit 1
fi

# 2. 检查/安装 uv
log_info "====== 步骤 2: 检查 uv 包管理器 ======"
if ! command -v uv &> /dev/null; then
    log_info "安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    # 重新加载 PATH
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    fi
else
    log_info "uv 已安装: $(uv --version)"
fi

# 3. 创建虚拟环境（如果不存在）
log_info "====== 步骤 3: 创建/激活虚拟环境 ======"
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    log_info "创建虚拟环境..."
    uv venv || python3 -m venv .venv
fi

# 激活虚拟环境
log_info "激活虚拟环境..."
source "$PROJECT_DIR/.venv/bin/activate"

# 4. 升级 pip
log_info "====== 步骤 4: 升级 pip ======"
python -m pip install --upgrade pip -q || true

# 5. 安装 PyTorch（特殊处理，可能有 CUDA）
log_info "====== 步骤 5: 安装 PyTorch ======"

# 检测 CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    log_info "检测到 CUDA $CUDA_VERSION，安装 GPU 版本 PyTorch..."
    retry 3 "pip install torch>=2.3.1 torchvision>=0.18.1 --index-url https://download.pytorch.org/whl/cu121 -q" || \
    retry 3 "pip install torch>=2.3.1 torchvision>=0.18.1 -q"
else
    log_info "未检测到 CUDA，安装 CPU 版本 PyTorch..."
    retry 3 "pip install torch>=2.3.1 torchvision>=0.18.1 -q"
fi

# 6. 安装其他依赖
log_info "====== 步骤 6: 安装项目依赖 ======"

# 使用 uv sync 或 pip install
if [ -f "pyproject.toml" ]; then
    log_info "使用 uv sync 安装依赖..."
    retry 3 "uv sync -q" || retry 3 "pip install -e . -q"
elif [ -f "requirements.txt" ]; then
    log_info "使用 requirements.txt 安装依赖..."
    retry 3 "pip install -r requirements.txt -q"
fi

# 7. 验证关键包
log_info "====== 步骤 7: 验证关键包安装 ======"
REQUIRED_PACKAGES="torch transformers langchain openai numpy pandas"
MISSING_PACKAGES=""

for pkg in $REQUIRED_PACKAGES; do
    if python -c "import $pkg" 2>/dev/null; then
        log_info "✓ $pkg"
    else
        log_warn "✗ $pkg 未安装"
        MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    log_info "重新安装缺失的包: $MISSING_PACKAGES"
    retry 3 "pip install$MISSING_PACKAGES -q"
fi

# 8. 配置环境变量
log_info "====== 步骤 8: 配置环境变量 ======"
if [ ! -f "$PROJECT_DIR/.env" ]; then
    if [ -f "$PROJECT_DIR/.env.example" ]; then
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        log_info "已创建 .env 文件，请填写你的 API 密钥"
    else
        log_warn "未找到 .env.example，创建默认 .env..."
        cat > "$PROJECT_DIR/.env" << 'EOF'
# Eye-LLM Environment Variables

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Qwen Configuration (阿里云)
QWEN_API_KEY=your-qwen-api-key-here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Replicate Configuration
REPLICATE_API_TOKEN=your-replicate-api-token-here

# Memory Configuration
SHORT_TERM_SIZE=5
LONG_TERM_MAX_SIZE=10000

# Attention Levels
ATTENTION_A=120
ATTENTION_B=60
ATTENTION_C=30
ATTENTION_D=15
ATTENTION_E=5
EOF
        log_info "已创建 .env 模板"
    fi
else
    log_info ".env 文件已存在"
fi

# 9. 创建必要的目录
log_info "====== 步骤 9: 创建必要目录 ======"
mkdir -p "$PROJECT_DIR/data/outputs/predictions"
mkdir -p "$PROJECT_DIR/data/outputs/heatmaps"
mkdir -p "$PROJECT_DIR/data/outputs/trajectories"
mkdir -p "$PROJECT_DIR/sam2/checkpoints"
mkdir -p "$PROJECT_DIR/logs"
log_info "目录结构创建完成"

# 10. 下载 SAM 2 模型（可选）
log_info "====== 步骤 10: 检查 SAM 2 模型 ======"
SAM2_MODEL="$PROJECT_DIR/sam2/checkpoints/sam2.1_hiera_small.pt"
if [ ! -f "$SAM2_MODEL" ]; then
    log_info "SAM 2 模型不存在，开始下载..."
    retry 3 "wget -O '$SAM2_MODEL' https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" || \
    log_warn "SAM 2 模型下载失败，可以稍后手动下载"
else
    log_info "SAM 2 模型已存在"
fi

# 11. 运行测试
log_info "====== 步骤 11: 运行基础测试 ======"
python -c "
import torch
import transformers
import langchain
import numpy as np
import pandas as pd
print('✓ 所有核心包导入成功')
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
" || log_warn "部分包导入失败，请手动检查"

# 12. 生成状态报告
log_info "====== 步骤 12: 生成状态报告 ======"
cat > "$PROJECT_DIR/logs/setup_status.txt" << EOF
Eye-LLM 环境配置报告
===================
配置时间: $(date)
Python 版本: $(python --version)
UV 版本: $(uv --version 2>/dev/null || echo "未安装")

虚拟环境: $PROJECT_DIR/.venv
环境配置文件: $PROJECT_DIR/.env

关键包版本:
$(pip list | grep -E "(torch|transformers|langchain|openai|numpy|pandas)")

下一步:
1. 编辑 $PROJECT_DIR/.env 填写 API 密钥
2. 运行测试: python -m pytest tests/ -v
3. 启动服务: python main.py
EOF

# 完成
echo ""
echo "=========================================="
echo -e "${GREEN}  环境配置完成！${NC}"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "  1. 编辑 .env 文件，填入你的 API 密钥:"
echo "     nano $PROJECT_DIR/.env"
echo ""
echo "  2. 激活虚拟环境:"
echo "     source $PROJECT_DIR/.venv/bin/activate"
echo ""
echo "  3. 运行测试:"
echo "     cd $PROJECT_DIR && python scripts/test_api_connection.py"
echo ""
echo "  4. 查看配置报告:"
echo "     cat $PROJECT_DIR/logs/setup_status.txt"
echo ""
echo "=========================================="

# 保持虚拟环境激活状态
exec "$SHELL"
