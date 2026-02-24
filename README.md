# Eye-LLM: Spatial Intent Prediction System

> **IROS 2026 - Multi-Modal Gaze Prediction with Memory, Topology, and LLM Reasoning**

Eye-LLM 是一个先进的空间意图预测系统，结合了大语言模型 (LLM)、计算机视觉和拓扑推理来预测复杂环境中的人类凝视模式和空间意图。

## 🌟 核心特性

- **双层记忆系统**: 短期记忆（滑动窗口）+ 长期记忆（统计信息）
- **拓扑推理引擎**: 基于环境拓扑的空间推理
- **多模态预测**: LLM 推理 + 多步序列预测
- **SAM 2 本地分割**: 支持本地和云端语义分割
- **像素级热力图**: 基于高斯模糊的眼动热力图
- **完整可视化**: 热力图、轨迹图、网络图

## 📦 安装

### 前置要求

- Python 3.11+
- uv (Python 包管理器)
- 阿里云 Qwen API Key

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/supa-2/IROS_Gaze_2026.git
cd IROS_Gaze_2026

# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

### SAM 2 本地部署（可选）

```bash
# 下载 SAM 2 模型
wget -O sam2/checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# 测试 SAM 2
python scripts/test_sam2_local.py --image data/R.jpg
```

## 🚀 快速开始

### 查看系统故事

```bash
# 展示 Eye-LLM 的完整故事
python scripts/demo_pipeline.py --mode story
```

### 完整演示流程

```bash
# 运行完整演示（包含系统概述、记忆、预测、可视化）
python scripts/demo_pipeline.py --mode full --image data/R.jpg

# 快速演示（仅记忆和预测）
python scripts/demo_pipeline.py --mode quick
```

### 眼动可视化

```bash
# 统一可视化流程（VLM 识别 + 热力图 + 轨迹图）
python scripts/unified_visualization.py --image data/R.jpg

# 使用已有 VLM 结果
python scripts/unified_visualization.py --image data/R.jpg --no-vlm
```

### SAM 2 本地分割

```bash
# 测试 SAM 2 本地分割
python scripts/test_sam2_local.py --image data/R.jpg

# 使用不同模型
python scripts/test_sam2_local.py --image data/R.jpg --model tiny
```

### 对比实验

```bash
# 运行所有实验（对比实验 + 消融实验）
python scripts/experiments/comparative_experiments.py --all

# 仅运行对比实验
python scripts/experiments/comparative_experiments.py --comparative
```

## 💻 Python API

### 基础使用

```python
from agent import EyeLLMAgent

# 初始化 Agent
agent = EyeLLMAgent(map_name='TH', use_new_architecture=True)

# 添加观测记录
agent.add_observation("TH-E01", attention_level='A')
agent.add_observation("TH-I-B01", attention_level='B')

# 预测下一个展品
prediction = agent.predict_next()
print(f"下一个: {prediction['prediction_name']}")
print(f"注意力等级: {prediction['attention_level']}")
print(f"置信度: {prediction['confidence']:.2f}")
```

### 序列预测

```python
# 预测未来 5 步
sequence = agent.predict_sequence(n_steps=5)
for i, step in enumerate(sequence):
    print(f"{i+1}. {step['prediction_name']} - {step['estimated_duration']}s")
```

### 可视化

```python
# 轨迹可视化
agent.visualize_trajectory()

# 热力图可视化
agent.visualize_heatmap()

# 像素级热力图
agent.visualize_pixel_heatmap(image_path='data/R.jpg')
```

### SAM 2 本地分割

```python
from skills.segmentation.sam2_local import SAM2LocalSegmenter

# 初始化分割器
segmenter = SAM2LocalSegmenter(model_size='small', device='cpu')

# 分割图像
exhibits = segmenter.segment_exhibits('data/R.jpg', min_area=5000)

for ex in exhibits:
    print(f"{ex['id']}: bbox={ex['bbox']}, score={ex['score']:.3f}")
```

## 📁 项目结构

```
iros_agent/
├── agent.py                    # 主 Agent 类
├── config.py                   # 全局配置
├── main.py                     # 入口文件
├── pyproject.toml              # 依赖管理
│
├── sam2/                       # SAM 2 本地部署
│   ├── checkpoints/            # 模型权重
│   └── sam2/                   # SAM 2 代码
│
├── skills/                     # 核心功能模块
│   ├── memory/                 # 记忆系统
│   │   ├── manager.py          # 记忆管理器
│   │   ├── short_term.py       # 短期记忆
│   │   └── long_term.py        # 长期记忆
│   │
│   ├── prediction/             # 预测引擎
│   │   ├── engine.py           # 预测引擎
│   │   ├── context_builder.py  # 上下文构建
│   │   ├── llm_reasoner.py     # LLM 推理
│   │   └── sequence_predictor.py # 序列预测
│   │
│   ├── segmentation/           # 语义分割
│   │   ├── segmenter.py        # Replicate API
│   │   └── sam2_local.py      # SAM 2 本地
│   │
│   ├── topology/               # 拓扑引擎
│   │   ├── graph_engine.py     # 图引擎
│   │   └── vlm_mapper.py       # VLM 映射
│   │
│   └── visualization/          # 可视化
│       ├── heatmap.py          # 网络热力图
│       ├── trajectory.py       # 轨迹图
│       ├── network.py          # 网络图
│       └── pixel_heatmap.py    # 像素热力图
│
├── scripts/                   # 脚本
│   ├── demo_pipeline.py        # 演示流程
│   ├── unified_visualization.py # 统一可视化
│   ├── test_sam2_local.py     # SAM 2 测试
│   └── experiments/           # 实验脚本
│
├── tests/                     # 测试
├── data/                      # 数据
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   └── outputs/               # 输出结果
│
└── docs/                      # 文档
    ├── IMPLEMENTATION_SUMMARY.md
    └── SAM2_LOCAL_DEPLOYMENT.md
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 测试特定模块
pytest tests/test_memory.py -v
pytest tests/test_prediction.py -v
```

## ⚙️ 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| QWEN_API_KEY | 阿里云 Qwen API Key | - |
| QWEN_BASE_URL | Qwen API 地址 | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| LLM_MODEL | LLM 模型名 | qwen-plus |
| VLM_MODEL | VLM 模型名 | qwen-vl-max-latest |

### 注意力等级

| 等级 | 时长 | 说明 |
|------|------|------|
| A | 120s | 深度关注 - 长时间仔细观看 |
| B | 60s | 中等关注 - 正常观看 |
| C | 30s | 一般关注 - 浏览式观看 |
| D | 15s | 快速浏览 - 短暂停留 |
| E | 5s | 一瞥 - 快速扫视 |

## 📊 输出示例

### 轨迹可视化
显示历史凝视路径，包含时间戳和注意力等级。

### 热力图可视化
基于凝视数据的空间注意力分布。

### 网络可视化
展品节点和连接权重的拓扑图。

### 像素级热力图
叠加在原图上的高斯模糊热力图。

## 🔬 实验结果

### 对比实验

| 模型 | 准确率 | Top-3 准确率 | 时长 MAE |
|------|--------|-------------|---------|
| Eye-LLM | 68.3% | 89.1% | 12.1s |
| 频率基线 | 32.5% | 54.2% | 28.3s |
| 随机基线 | 15.2% | 38.7% | 45.6s |

### 消融实验

| 配置 | 准确率 | Δ |
|------|--------|-----|
| 完整模型 | 68.3% | - |
| 无记忆 | 53.1% | -15.2% |
| 无拓扑 | 55.5% | -12.8% |
| 无 CoT | 60.5% | -7.8% |

## 📝 文档

- [实施总结](docs/IMPLEMENTATION_SUMMARY.md)
- [SAM 2 本地部署指南](docs/SAM2_LOCAL_DEPLOYMENT.md)
- [项目配置文档](CLAUDE.md)

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 📧 联系方式

如有问题或合作意向，请联系：
- GitHub: @supa-2

---

**IROS 2026 - Eye-LLM 空间意图预测系统** 👁️🔍
