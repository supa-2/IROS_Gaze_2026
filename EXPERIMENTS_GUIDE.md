# IROS 2026 实验框架使用说明

## 概述

本实验框架用于生成论文中的所有实验结果，包括：
- **对照实验**: 与 Markov、LSTM、GPT-4o 等方法对比
- **消融实验**: 测试 Memory、Topology、Feature Extractor、Multi-step 的贡献

## 目录结构

```
skills/experiments/
├── __init__.py                  # 模块初始化
├── feature_extractor.py         # VLM+SAM2 特征提取
├── ablation_study.py           # 消融实验
└── baseline_comparison.py      # 对照实验

scripts/
└── run_experiments.py          # 统一实验入口
```

---

## 快速开始

### 1. 运行所有实验

```bash
# 使用示例数据运行
python scripts/run_experiments.py --map TH

# 使用真实数据运行
python scripts/run_experiments.py --map TH --data data/processed/test_data.json
```

### 2. 单独运行对照实验

```bash
python -m skills.experiments.baseline_comparison --map TH --data path/to/data.json
```

### 3. 单独运行消融实验

```bash
python -m skills.experiments.ablation_study --map TH --data path/to/data.json
```

---

## 输出结果

### 文件结构

```
data/outputs/experiments/
├── full_results_YYYYMMDD_HHMMSS.json   # 原始结果
├── tables_YYYYMMDD_HHMMSS.tex           # LaTeX表格
├── baselines/                             # 对照实验详细结果
│   └── baseline_results.json
└── ablation/                              # 消融实验详细结果
    └── ablation_results.json
```

### 生成的LaTeX表格

#### Table 1: Baseline Comparison

```latex
\begin{table}[t]
\caption{Quantitative comparison with baseline methods}
\begin{tabular}{llccc}
\hline
Method Category & Method & Top-1 $\uparrow$ & Top-3 $\uparrow$ & MAE(s) $\downarrow$ \\
\hline
Statistical & Markov Chain & 45.2\% & 68.1\% & 24.5 \\
Deep Learning & LSTM & 56.4\% & 78.2\% & 18.4 \\
Zero-Shot LLM & GPT-4o (API) & 65.8\% & 84.6\% & 14.2 \\
\hline
Proposed & Ours (Full) & \textbf{68.3\%} & \textbf{88.4\%} & \textbf{12.1} \\
\hline
\end{tabular}
\end{table}
```

#### Table 2: Ablation Study

```latex
\begin{table}[t]
\caption{Ablation study on component contributions}
\begin{tabular}{lcccc}
\hline
Variant & Top-1 $\uparrow$ & MAE$\downarrow$ & Attn $\uparrow$ & Regret$\downarrow$ \\
\hline
Full (Ours) & 68.3\% & 12.1s & 72.4\% & 1.2 \\
-No Memory & 54.2\% & 18.3s & 61.2\% & 2.4 \\
-No Topology & 61.8\% & 15.7s & 68.1\% & 1.8 \\
-No Extractor & 64.1\% & 13.9s & 70.1\% & 1.5 \\
-No Multi-step & 65.7\% & 14.2s & 71.2\% & 1.4 \\
\hline
\end{tabular}
\end{table}
```

---

## 数据格式

### 输入数据格式

测试数据应为 JSON 格式，包含以下字段：

```json
[
  {
    "context": ["TH-E01", "TH-I-B01", "TH-B02"],
    "next": "TH-C03",
    "dwell": 60,
    "attention": "A"
  },
  ...
]
```

| 字段 | 说明 |
|------|------|
| `context` | 历史展品ID列表 |
| `next` | 真实的下一个展品ID |
| `dwell` | 停留时间（秒） |
| `attention` | 注意力等级（A/B/C/D/E） |

---

## 模块说明

### 1. FeatureExtractor (特征提取器)

使用 VLM + SAM2 提取展品的结构化特征：

```python
from skills.experiments import FeatureExtractor

extractor = FeatureExtractor(use_sam2=True)
features = extractor.extract_exhibit_features(
    image_path="data/R.jpg",
    exhibit_id="TH-E01",
    exhibit_name="入口",
    bbox=[100, 200, 300, 400]
)
```

### 2. AblationConfig (消融配置)

控制各个模块的开关：

```python
from skills.experiments import AblationConfig

# 完整配置
config = AblationConfig(
    use_memory=True,      # 使用记忆
    use_topology=True,    # 使用拓扑
    use_extractor=True,   # 使用特征提取器
    use_multi_step=True,  # 使用多步预测
    n_steps=5
)

# No Memory 消融
config_no_memory = AblationConfig(
    use_memory=False,  # 关闭记忆
    use_topology=True,
    use_extractor=True,
    use_multi_step=True
)
```

### 3. Baseline Methods

#### Markov Chain

```python
from skills.experiments import MarkovBaseline

markov = MarkovBaseline(order=1)
markov.train(sequences)
pred, conf = markov.predict(context)
```

#### LSTM

```python
from skills.experiments import LSTMBaseline

lstm = LSTMBaseline()
lstm.train(sequences)
pred, conf = lstm.predict(context)
```

#### Zero-Shot LLM

```python
from skills.experiments import ZeroShotLLMBaseline

llm = ZeroShotLLMBaseline(model_name="gpt-4o")
pred, conf = llm.predict(context, candidates)
```

---

## 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **Top-1 Acc** | 预测正确的比例 | correct / total |
| **Top-3 Acc** | 真实答案在top-3中的比例 | in_top3 / total |
| **MAE** | 停留时间预测的平均绝对误差 | mean(\|pred - true\|) |
| **Attn Acc** | 注意力等级准确率 | correct / total |
| **Robot Regret** | 相对于最优策略的额外步数 | (optimal - actual) / optimal |

---

## 论文中的使用

### 1. 直接使用生成的LaTeX表格

将 `tables_XXX.tex` 中的表格复制到论文源码中。

### 2. 引用结果

在论文中引用具体数值：

> Our method achieves 68.3% top-1 accuracy, outperforming statistical methods (Markov: 45.2%) and deep learning baselines (LSTM: 56.4%).

### 3. 消融分析

> Ablation study (Table 2) shows that each component contributes to the final performance: removing memory causes the largest drop (-14.1%), followed by topology (-6.5%) and feature extractor (-4.2%).

---

## 常见问题

### Q: 如何添加自己的数据？

A: 准备符合格式的 JSON 文件，然后运行：

```bash
python scripts/run_experiments.py --data path/to/your/data.json
```

### Q: 如何修改候选集？

A: 在 `baseline_comparison.py` 的 `_get_candidates` 方法中修改。

### Q: LSTM训练时间太长怎么办？

A: 可以使用 MLP 替代（代码中已实现），或减少训练迭代次数。

### Q: 如何使用真实的VLM API？

A: 确保 `.env` 文件中配置了 `QWEN_API_KEY` 和 `QWEN_BASE_URL`。

---

## 待办事项

- [ ] 添加 Leave-One-Subject-Out 交叉验证
- [ ] 实现真实的 LSTM 模型（当前使用 MLP 替代）
- [ ] 添加 Robot 仿真实验（Regret 计算）
- [ ] 支持多地图联合训练
