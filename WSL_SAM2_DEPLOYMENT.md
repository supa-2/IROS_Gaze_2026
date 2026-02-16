# SAM 2 WSL 部署教程

本教程指导你在 WSL (Windows Subsystem for Linux) 上部署 SAM 2 模型用于图像分割。

---

## 前提条件

1. **WSL 2** 已安装 (推荐 Ubuntu 22.04 或 24.04)
2. **Python 3.11+**
3. **CUDA 支持** (如果使用 GPU)

---

## 步骤 1: 更新 WSL 系统

```bash
# 更新软件包列表
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y build-essential git wget curl python3 python3-pip python3-venv

# 安装 Python 开发依赖（编译某些包需要）
sudo apt install -y python3-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

---

## 步骤 2: 克隆项目

```bash
# 进入工作目录
cd ~

# 克隆你的项目
git clone https://github.com/supa-2/IROS_Gaze_2026.git
cd IROS_Gaze_2026
```

---

## 步骤 3: 创建虚拟环境

```bash
# 创建 Python 虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 升级 pip
pip install --upgrade pip
```

---

## 步骤 4: 安装 PyTorch (CUDA 版本)

### 4.1 检查 CUDA 版本

```bash
nvidia-smi
```

查看输出中的 `CUDA Version`，例如 `12.4`。

### 4.2 安装对应版本的 PyTorch

访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合你 CUDA 版本的安装命令。

**常见版本示例**：

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 版本 (无 GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

验证安装：

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 步骤 5: 安装 SAM 2 依赖

```bash
# 安装 SAM 2 官方包
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# 安装项目其他依赖
pip install -r <(cat <<'EOF'
langchain>=1.2.3
langchain-openai>=1.1.7
langgraph>=1.0.5
networkx>=3.6.1
python-dotenv>=1.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
pillow>=10.2.0
opencv-python>=4.9.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
transformers>=4.0.0
openpyxl>=3.1.0
EOF
)
```

---

## 步骤 6: 下载 SAM 2 模型

### 方法 A: 直接下载 (推荐)

```bash
# 创建模型目录
mkdir -p models/sam2

# 下载 SAM 2 tiny 模型 (~40MB)
wget https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2-hiera-tiny.pt -P models/sam2/

# 或者下载 small 模型 (~140MB，精度更高)
# wget https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2-hiera-small.pt -P models/sam2/

# 验证下载
ls -lh models/sam2/
```

### 方法 B: 在 Windows 下载后复制

1. 在 Windows 浏览器访问: https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2-hiera-tiny.pt
2. 下载到 Windows 后，复制到 WSL:
   ```bash
   # 在 WSL 中执行（假设下载到 Windows Downloads 文件夹）
   cp /mnt/c/Users/你的用户名/Downloads/sam2-hiera-tiny.pt models/sam2/
   ```

---

## 步骤 7: 创建 SAM 2 测试脚本

```bash
# 创建测试脚本
cat > test_sam2.py << 'EOF'
#!/usr/bin/env python3
"""SAM 2 本地测试脚本"""

import torch
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def test_sam2():
    """测试 SAM 2 模型加载和推理"""

    print("=" * 60)
    print("SAM 2 本地部署测试")
    print("=" * 60)

    # 1. 检查 CUDA
    print(f"\n[1/4] 检查 CUDA...")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("  Using CPU")
        device = "cpu"

    # 2. 加载模型
    print(f"\n[2/4] 加载 SAM 2 模型...")
    model_path = "models/sam2/sam2-hiera-tiny.pt"

    try:
        sam2_model = build_sam2(
            config_file="sam2/configs/sam2-hiera-t.yaml",  # 需要下载配置文件
            ckpt_path=model_path,
            device=device,
        )
        print(f"  ✓ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        print("\n提示: 如果缺少配置文件，请运行:")
        print("  git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo")
        return

    # 3. 创建预测器
    print(f"\n[3/4] 创建预测器...")
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"  ✓ 预测器创建成功")

    # 4. 测试推理
    print(f"\n[4/4] 测试推理...")

    # 创建测试图片
    test_image_path = "test_image.png"
    create_test_image(test_image_path)

    # 设置图片
    image = np.array(Image.open(test_image_path))
    predictor.set_image(image)
    print(f"  ✓ 图片已设置: {image.shape}")

    # 自动分割
    print(f"\n运行自动分割...")
    masks, scores, logits = predictor.predict(
        point_coords=None,  # 自动模式
        point_labels=None,
        box=None,
        multimask_output=True,
    )

    print(f"  ✓ 分割完成!")
    print(f"    - 检测到 {len(masks)} 个掩码")
    print(f"    - 分数范围: {scores.min():.3f} ~ {scores.max():.3f}")

    # 保存结果
    save_masks(masks, test_image_path)

    print("\n" + "=" * 60)
    print("测试完成！结果保存在 data/outputs/masks/")
    print("=" * 60)


def create_test_image(path):
    """创建测试图片"""
    import os
    os.makedirs("data/outputs/masks", exist_ok=True)

    # 创建一个简单的测试图片（带形状）
    from PIL import ImageDraw

    img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(img)

    # 画几个形状
    draw.ellipse([100, 100, 300, 300], fill='red')
    draw.rectangle([350, 150, 480, 350], fill='blue')
    draw.polygon([(200, 350), (300, 450), (100, 450)], fill='green')

    img.save(path)
    print(f"  创建测试图片: {path}")


def save_masks(masks, image_path):
    """保存分割掩码"""
    import os
    from PIL import Image

    output_dir = "data/outputs/masks"
    os.makedirs(output_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        mask_img.save(mask_path)
        print(f"  保存掩码: {mask_path}")


if __name__ == "__main__":
    test_sam2()
EOF

# 运行测试
python3 test_sam2.py
```

---

## 步骤 8: 获取 SAM 2 配置文件

如果测试脚本提示缺少配置文件：

```bash
# 克隆 SAM 2 仓库
cd ~
git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo

# 复制配置到项目目录
cp -r sam2_repo/sam2/configs ~/IROS_Gaze_2026/
cp sam2_repo/sam2/*.py ~/IROS_Gaze_2026/sam2/ 2>/dev/null || true
```

---

## 步骤 9: 配置环境变量

```bash
# 创建 .env 文件
cat > .env << 'EOF'
# OpenAI API (用于LLM)
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# SAM 2 配置
SAM2_MODEL_PATH=models/sam2/sam2-hiera-tiny.pt
SAM2_DEVICE=cuda  # 或 cpu

# 其他配置
SHORT_TERM_SIZE=5
LONG_TERM_MAX_SIZE=10000
EOF

# 确保 .env 在 .gitignore 中
echo ".env" >> .gitignore
```

---

## 步骤 10: 验证完整部署

```bash
# 激活虚拟环境
source .venv/bin/activate

# 测试导入
python3 -c "
import torch
import sys
sys.path.insert(0, '.')
from skills.prediction.engine import PredictionEngine
from config import Config
print('✓ 所有模块导入成功')
"

# 测试 SAM 2
python3 test_sam2.py
```

---

## 常见问题解决

### Q1: CUDA out of memory

```bash
# 使用 CPU 模式
export SAM2_DEVICE=cpu
# 或在代码中设置
torch.cuda.empty_cache()
```

### Q2: 模型加载失败

```bash
# 检查模型文件
ls -lh models/sam2/

# 重新下载
rm models/sam2/*.pt
wget https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2-hiera-tiny.pt -P models/sam2/
```

### Q3: 配置文件缺失

```bash
# 克隆完整仓库获取配置
cd ~
git clone https://github.com/facebookresearch/segment-anything-2.git
cp -r segment-anything-2/sam2 ~/IROS_Gaze_2026/
```

### Q4: Windows 路径访问

```bash
# 访问 Windows 文件系统
cd /mnt/c/Users/你的用户名/...

# 从 Windows 复制文件到 WSL
cp /mnt/c/path/to/file.png .
```

---

## 完整部署检查清单

- [ ] WSL 2 已安装并更新
- [ ] Python 3.11+ 已安装
- [ ] 虚拟环境已创建并激活
- [ ] PyTorch (CUDA版本) 已安装
- [ ] SAM 2 包已安装
- [ ] 模型文件已下载 (`models/sam2/sam2-hiera-tiny.pt`)
- [ ] 配置文件已获取 (`sam2/configs/`)
- [ ] 测试脚本运行成功
- [ ] `.env` 文件已配置

---

## 下一步

部署完成后，你可以：

1. **运行完整系统**
   ```bash
   python3 agent.py
   ```

2. **使用 SAM 2 分割图片**
   ```python
   from skills.segmentation.segmenter import SemanticSegmenter
   segmenter.segment_image("data/R.jpg")
   ```

3. **训练微调模型**
   ```bash
   llamafactory-cli train config/qwen2_lora_sft.yaml
   ```

---

*文档版本: 1.0*
*更新时间: 2025-02-14*
