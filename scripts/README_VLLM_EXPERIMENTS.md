# vLLM Experiments - 使用本地微调模型运行实验

本目录包含用于在有 GPU 的服务器上运行实验的脚本，使用您的微调模型 `qwen2.5-32b-int4`。

## 前置要求

### 1. 硬件要求
- GPU 内存: 至少 24GB (推荐 32GB+)
- 模型大小: ~20GB (int4 量化)

### 2. 软件安装

```bash
# 安装 vLLM
pip install vllm

# 或使用 GPU 版本
pip install vllm-gpu
```

### 3. 模型位置

确保您的微调模型位于:
```
/home/g/models/qwen2.5-32b-int4
```

## 可用脚本

### 1. `run_vllm_ablation.py` - 消融实验 (推荐)

使用您的微调模型运行完整的消融实验，测试各组件贡献。

```bash
# 基本运行
python scripts/run_vllm_ablation.py

# 指定模型路径
python scripts/run_vllm_ablation.py --model /path/to/model

# 指定测试数据
python scripts/run_vllm_ablation.py --data /path/to/test_data.json

# 指定地图
python scripts/run_vllm_ablation.py --map TH  # 或 OS
```

**输出**:
- `data/outputs/vllm_ablation/ablation_results.json` - JSON 格式结果
- 终端打印 LaTeX 表格 (可直接用于论文)

**测试的配置**:
1. **Full (Ours)** - 完整模型，所有组件启用
2. **No-Memory** - 移除历史记忆
3. **No-Topology** - 移除空间拓扑约束
4. **No-Extractor** - 移除特征提取器
5. **No-Multi-step** - 仅单步预测

**评估指标**:
- Top-1 Accuracy: 下一位置预测准确率
- Top-3 Accuracy: 前三个预测中包含真实值
- MAE: 停留时间预测平均绝对误差 (秒)
- Attn Accuracy: 注意力等级预测准确率

### 2. `run_vllm_experiments.py` - 基础实验

运行基本实验评估模型性能。

```bash
python scripts/run_vllm_experiments.py --model /home/g/models/qwen2.5-32b-int4
```

## 数据格式

### 测试数据格式 (JSON)

```json
[
  {
    "context": ["TH-E01", "TH-I-B01", "TH-B02"],
    "current": "TH-B02",
    "next": "TH-C03",
    "history": [
      {"id": "TH-E01", "name": "入口", "level": "B", "duration": 60},
      {"id": "TH-I-B01", "name": "导览图", "level": "A", "duration": 120}
    ],
    "dwell": 60,
    "attention": "A"
  }
]
```

### 字段说明
- `context`: 完整的访问轨迹
- `current`: 当前位置
- `next`: 下一个位置 (ground truth)
- `history`: 历史记录列表
- `dwell`: 停留时间 (秒)
- `attention`: 注意力等级 (A/B/C/D/E)

## 输出示例

### LaTeX 表格输出

```
\begin{table}[t]
\centering
\caption{Ablation study with fine-tuned Qwen2.5-32B-int4}
\label{tab:ablation}
\begin{tabular}{lccccc}
\hline
Variant & Top-1 $\uparrow$ & Top-3 $\uparrow$ & MAE$\downarrow$ & Attn $\uparrow$ \\
\hline
Full (Ours) & 68.3% & 88.4% & 12.1s & 72.4% \\
-No-Memory & 62.1% (-6.2) & 83.5% (-4.9) & 15.8s (+3.7) & 68.2% (-4.2) \\
-No-Topology & 58.4% (-9.9) & 79.1% (-9.3) & 18.2s (+6.1) & 65.1% (-7.3) \\
\hline
\end{tabular}
\end{table}
```

### 文本表格输出

```
Variant              Top-1      Top-3       MAE       Attn
--------------------------------------------------------------
Full (Ours)          68.3%      88.4%      12.1s      72.4%
No-Memory            62.1%      83.5%      15.8s      68.2%
No-Topology          58.4%      79.1%      18.2s      65.1%
```

## 在服务器上运行

### 1. 传输脚本和数据

```bash
# 将项目复制到服务器
rsync -avz /path/to/IROS_Gaze_2026/ user@server:/path/to/destination/

# 或仅复制必要文件
scp scripts/run_vllm_ablation.py user@server:/path/to/scripts/
```

### 2. SSH 登录到服务器

```bash
ssh user@server
cd /path/to/IROS_Gaze_2026
```

### 3. 激活环境并运行

```bash
# 激活虚拟环境 (如果使用)
source venv/bin/activate

# 运行消融实验
python scripts/run_vllm_ablation.py --model /home/g/models/qwen2.5-32b-int4 --map TH
```

### 4. 查看结果

结果将保存在 `data/outputs/vllm_ablation/` 目录下。

```bash
cat data/outputs/vllm_ablation/ablation_results.json
```

## 故障排查

### 问题: GPU 内存不足

```bash
# 降低 GPU 内存利用率
python scripts/run_vllm_ablation.py --model /home/g/models/qwen2.5-32b-int4

# 然后修改脚本中的 gpu_memory_utilization 参数 (0.9 -> 0.7)
```

### 问题: vLLM 未安装

```bash
pip install vllm
```

### 问题: 模型路径错误

```bash
# 检查模型是否存在
ls -la /home/g/models/qwen2.5-32b-int4

# 使用正确的路径
python scripts/run_vllm_ablation.py --model /correct/path/to/model
```

## 与论文的对应

| 脚本配置 | 论文表格列 | 说明 |
|---------|-----------|------|
| Full (Ours) | Ours | 完整方法 |
| No-Memory | w/o Memory | 移除记忆模块 |
| No-Topology | w/o Topology | 移除空间拓扑 |
| No-Extractor | w/o Feature | 移除特征提取 |
| No-Multi-step | w/o Multi-step | 单步预测 |

## 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--model` | `/home/g/models/qwen2.5-32b-int4` | 模型路径 |
| `--data` | `None` | 测试数据路径 (None=使用模拟数据) |
| `--map` | `TH` | 地图名称 (TH/OS) |

## 联系

如有问题，请检查:
1. vLLM 版本是否兼容
2. 模型文件是否完整
3. GPU 内存是否足够
4. 数据格式是否正确
