# WSL + SAM 2 完整部署指南

## 快速部署（推荐）

### 方式一：使用在线 API（最简单，无需 GPU）

**在 WSL 终端执行：**

```bash
# 1. 复制项目到 WSL
cp -r /mnt/c/Users/你的用户名/IROS_Gaze_2026 ~/IROS_Gaze_2026
cd ~/IROS_Gaze_2026

# 2. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# 3. 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate
uv pip install replicate pillow numpy

# 4. 设置 API Token
export REPLICATE_API_TOKEN=r8_OeqoF6RNCfM94oc2BzDO1KEAegiu9oD0MGY9N

# 5. 调用 SAM 2
python scripts/call_sam2.py --image data/R.jpg --output data/outputs/masks/
```

---

### 方式二：本地模型部署（需要 GPU，速度快）

**在 WSL 终端执行：**

```bash
# 1. 复制项目到 WSL
cp -r /mnt/c/Users/你的用户名/IROS_Gaze_2026 ~/IROS_Gaze_2026
cd ~/IROS_Gaze_2026

# 2. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# 3. 创建虚拟环境
uv venv
source .venv/bin/activate

# 4. 安装 PyTorch (CUDA 12.1)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 安装 SAM 2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# 6. 下载模型
mkdir -p models/sam2
wget https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2-hiera-tiny.pt -P models/sam2/

# 7. 获取配置文件
cd ~
git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo
cp -r sam2_repo/sam2 ~/IROS_Gaze_2026/
cd ~/IROS_Gaze_2026

# 8. 测试调用
python scripts/call_sam2.py --image data/R.jpg --output data/outputs/masks/ --mode local
```

---

## 调用示例

### Python 代码中调用

```python
from scripts.call_sam2 import call_sam2_local, call_sam2_online

# 方式 1: 使用在线 API
result = call_sam2_online(
    image_path="data/R.jpg",
    api_token="your_token"
)

# 方式 2: 使用本地模型
result = call_sam2_local(
    image_path="data/R.jpg",
    model_path="models/sam2/sam2-hiera-tiny.pt",
    device="cuda"  # 或 "cpu"
)

# 结果格式
print(result["masks"])      # 掩码列表
print(result["boxes"])      # 边界框列表 [(x1, y1, x2, y2), ...]
print(result["centers"])    # 中心点列表 [(x, y), ...]
print(result["scores"])     # 置信度列表
```

### 命令行调用

```bash
# 自动模式（有本地模型就用本地，没有就用在线）
python scripts/call_sam2.py --image data/R.jpg --output data/outputs/masks/

# 强制使用本地模型
python scripts/call_sam2.py --image data/R.jpg --mode local --device cuda

# 强制使用在线 API
python scripts/call_sam2.py --image data/R.jpg --mode online
```

---

## 常见问题

### Q: 在线 API 是免费的吗？

A: Replicate API 有免费额度，但有速率限制。大量调用建议部署本地模型。

### Q: 本地模型需要多少显存？

A: sam2-hiera-tiny 约需 2GB 显存即可运行。

### Q: 没有图片怎么测试？

A: 系统会创建测试图片，或使用 data/R.jpg（如果有）。

### Q: WSL 中访问 Windows 文件

A: 使用 `/mnt/c/` 前缀：
```bash
cp /mnt/c/Users/xxx/IROS_Gaze_2026 ~/IROS_Gaze_2026
```

---

## 推荐配置

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 测试/偶用 | 在线 API | 无需部署，开箱即用 |
| 频繁使用 | 本地模型 | 速度快，无 API 限制 |
| 无 GPU | 在线 API 或 CPU 模式 | 灵活选择 |
| 有 GPU | 本地模型 | 速度最快 |

---

*最后更新: 2025-02-14*
