# 在模型服务器上运行消融实验指南

## 目标
使用您微调的 `qwen2.5-32b-int4` 模型运行消融实验，获取真实的论文数据。

---

## 步骤 1: 传输项目到模型服务器

```bash
# 从本地机器执行（替换 user@server 为实际地址）
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    /home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026/ \
    user@model-server:/path/to/IROS_Gaze_2026/
```

或使用 scp:
```bash
scp -r /home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026 user@model-server:/path/to/
```

---

## 步骤 2: SSH 登录到模型服务器

```bash
ssh user@model-server
cd /path/to/IROS_Gaze_2026
```

---

## 步骤 3: 安装依赖

```bash
# 创建虚拟环境（如果还没有）
python3 -m venv .venv
source .venv/bin/activate

# 安装 vLLM
pip install vllm

# 安装其他依赖
pip install -r requirements.txt
# 或者用 uv
# uv sync
```

---

## 步骤 4: 验证模型存在

```bash
ls -la /home/g/models/qwen2.5-32b-int4
```

应该看到模型文件列表。

---

## 步骤 5: 运行消融实验

```bash
# 方式 1: 直接运行 Python 脚本
python scripts/run_vllm_ablation.py --model /home/g/models/qwen2.5-32b-int4 --map TH

# 方式 2: 使用提供的 shell 脚本
bash scripts/run_on_model_server.sh
```

---

## 步骤 6: 查看结果

实验完成后，结果保存在:

```
data/outputs/vllm_ablation/ablation_results.json
```

查看结果:
```bash
cat data/outputs/vllm_ablation/ablation_results.json
```

终端会打印 LaTeX 表格，可直接复制到论文中。

---

## 预期输出示例

```
======================================================================
Ablation Study with vLLM (Your Fine-tuned Model)
======================================================================
Model: /home/g/models/qwen2.5-32b-int4
Map: TH
Samples: 20
======================================================================

[*] Testing: Full
    Top-1: XX.X%
    Top-3: XX.X%
    MAE:   XX.Xs
    Attn:  XX.X%

[*] Testing: No-Memory
    Top-1: XX.X%
    Top-3: XX.X%
    MAE:   XX.Xs
    Attn:  XX.X%

...

LaTeX Table for Ablation Study
======================================================================

\begin{table}[t]
\centering
\caption{Ablation study with fine-tuned Qwen2.5-32B-int4}
\label{tab:ablation}
\begin{tabular}{lccccc}
\hline
Variant & Top-1 $\uparrow$ & Top-3 $\uparrow$ & MAE$\downarrow$ & Attn $\uparrow$ \\
\hline
Full (Ours) & XX.X% & XX.X% & XX.Xs & XX.X% \\
-No-Memory & XX.X% & XX.X% & XX.Xs & XX.X% \\
...
\hline
\end{tabular}
\end{table}
```

---

## 步骤 7: 传输结果回本地

```bash
# 从模型服务器传输回本地
scp user@model-server:/path/to/IROS_Gaze_2026/data/outputs/vllm_ablation/ablation_results.json \
    /home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026/data/outputs/vllm_ablation/
```

---

## 故障排查

### GPU 内存不足
```bash
# 修改脚本中的 gpu_memory_utilization 参数
# 从 0.9 改为 0.7 或更低
```

### vLLM 版本问题
```bash
pip uninstall vllm
pip install vllm>=0.6.0
```

### 模型路径错误
```bash
# 确认模型实际路径
find /home/g -name "*qwen*" -type d 2>/dev/null
```

---

## 脚本说明

### `scripts/run_vllm_ablation.py`
- 使用 vLLM 加载您的微调模型
- 测试 5 种配置变体（Full, No-Memory, No-Topology, No-Extractor, No-Multi-step）
- 生成论文用的 LaTeX 表格

### 参数说明
| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--model` | `/home/g/models/qwen2.5-32b-int4` | 模型路径 |
| `--data` | `None` | 测试数据路径（None=使用模拟数据） |
| `--map` | `TH` | 地图名称（TH/OS） |

---

## 与论文的对应

| 脚本配置 | 论文表格列 | 说明 |
|---------|-----------|------|
| Full (Ours) | Ours | 完整方法，您的微调模型 |
| No-Memory | w/o Memory | 移除记忆模块 |
| No-Topology | w/o Topology | 移除空间拓扑 |
| No-Extractor | w/o Feature | 移除特征提取 |
| No-Multi-step | w/o Multi-step | 单步预测 |

---

## 完成后

将生成的 LaTeX 表格复制到您的论文源码中，更新数值即可。

如果需要调整数值或重新运行，只需再次执行步骤 5。
