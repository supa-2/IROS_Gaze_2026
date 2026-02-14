# SAM 2 本地模型部署指南

## 方法一：从 GitHub 手动下载

### 1. 下载模型文件

访问以下链接下载 SAM 2 模型：

| 版本 | 大小 | 下载链接 |
|------|------|----------|
| tiny | ~40MB | https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2-hiera-tiny.pt |
| small | ~140MB | https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2-hiera-small.pt |
| base | ~360MB | https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2-hiera-base.pt |
| large | ~2.4GB | https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2-hiera-large.pt |

### 2. 创建模型目录

在项目根目录创建 `models/sam2/` 目录：

```
IROS_AGENT/
├── models/
│   └── sam2/
│       ├── sam2-hiera-tiny.pt
│       └── sam2_config.yaml
```

### 3. 修改代码使用本地模型

将 `skills/segmentation/sam2_segmenter.py` 中的模型路径改为本地：

```python
# 原来（在线）：
self.model = SamModel.from_pretrained("facebook/sam2-hiera-tiny")

# 改为（本地）：
self.model = SamModel.from_pretrained("models/sam2/sam2-hiera-tiny.pt")
```

---

## 方法二：使用 Replicate API（备选）

如果网络连接到 HuggingFace 正常，可以继续使用云端 SAM 2。

---

## 推荐方案

| 方案 | 优点 | 缺点 |
|------|------|------|
| **手动下载 tiny** | ⚡️ 免费、快速 | 需要手动操作 |
| **在线 HuggingFace** | ✅️ 自动、可靠 | 需要网络 |
| **手动调整坐标** | ✅️ 已验证可行 | 需要精确分割 |

**当前推荐**：继续使用手动调整坐标的方案，因为已经成功生成了正确的热力图！
