# Eye-LLM 实施总结

## 完成日期
2025-02-24

## 完成的工作

### 1. SAM 2 模型 ✅
- SAM 2 已通过 `sam2/` 目录安装
- 可通过 Replicate API 使用云端分割服务
- 支持本地和云端两种模式

### 2. 修复热力图和轨迹图可视化 ✅

#### 修复的问题：
1. **`pixel_heatmap.py`**: 添加了 `region_id` 属性到 `FixationPoint`，支持与轨迹图的数据共享
2. **`agent.py`**: 修复了 `visualize_pixel_heatmap()` 方法，正确调用 `visualize_side_by_side()` 或 `visualize_overlay()`
3. **创建统一可视化脚本**: `scripts/unified_visualization.py`

#### 新增的可视化功能：
- VLM 展品识别与边界框绘制
- 像素级眼动热力图（高斯模糊）
- 眼动轨迹图（序号标注，点大小反映时长）
- 热力图+轨迹图叠加
- 并排对比视图

### 3. 实验设计方案 ✅

#### 创建的文件：`scripts/experiments/comparative_experiments.py`

包含以下实验：

**对比实验 (Comparative Experiments)**：
- Eye-LLM vs 频率基线
- Eye-LLM vs 随机基线
- Eye-LLM vs 拓扑基线

**消融实验 (Ablation Studies)**：
- 完整模型
- 无记忆系统
- 无拓扑约束
- 无思维链推理
- 仅记忆模式

**评估指标**：
- 准确率 (Accuracy)
- Top-K 准确率
- 平均绝对误差 (MAE) - 用于时长预测
- 序列相似度 (Sequence Similarity) - 基于编辑距离

### 4. 完整展示流程 ✅

#### 创建的文件：`scripts/demo_pipeline.py`

三种演示模式：
1. **`--mode full`**: 完整演示（系统概述 → 记忆 → 预测 → 可视化 → 报告）
2. **`--mode quick`**: 快速演示（记忆 + 预测）
3. **`--mode story`**: 故事模式（讲述 Eye-LLM 的完整故事）

## 新增文件列表

```
scripts/
├── unified_visualization.py          # 统一眼动可视化脚本
├── demo_pipeline.py                  # 演示流程脚本
└── experiments/
    └── comparative_experiments.py    # 对比实验和消融实验

skills/visualization/
├── pixel_heatmap.py                  # 修复：添加 region_id 属性
└── gaze_trajectory.py                # 现有：与 pixel_heatmap 配合

agent.py                              # 修复：visualize_pixel_heatmap 方法
```

## 使用方式

### 1. 运行故事模式（快速了解系统）
```bash
python scripts/demo_pipeline.py --mode story
```

### 2. 运行完整演示
```bash
python scripts/demo_pipeline.py --mode full --map TH --image data/R.jpg
```

### 3. 运行统一可视化
```bash
# 使用 VLM 识别
python scripts/unified_visualization.py --image data/R.jpg

# 使用已有 VLM 结果
python scripts/unified_visualization.py --image data/R.jpg --no-vlm
```

### 4. 运行对比实验
```bash
# 运行所有实验
python scripts/experiments/comparative_experiments.py --all

# 仅运行对比实验
python scripts/experiments/comparative_experiments.py --comparative

# 仅运行消融实验
python scripts/experiments/comparative_experiments.py --ablation
```

## 故事讲述框架

为 IROS 投稿设计的故事线：

1. **问题 (The Problem)**
   - 展厅中的用户行为预测挑战
   - 空间布局、语义连接、个人记忆的复杂交互

2. **解决方案 (The Solution)**
   - 空间记忆 (Spatial Memory)
   - 拓扑约束 (Topological Constraints)
   - 语义推理 (Semantic Reasoning)

3. **创新点 (The Innovation)**
   - 双层记忆架构
   - 空间-语义融合
   - 注意力建模
   - 可解释预测

4. **结果 (The Results)**
   - 对比实验数据
   - 消融实验分析

5. **影响 (The Impact)**
   - 应用场景
   - 未来方向

## 注意事项

1. **API 配置**：确保 `.env` 文件中配置了正确的 API 密钥
   - QWEN_API_KEY: 阿里云 Qwen API
   - REPLICATE_API_TOKEN: SAM 2 云端分割（可选）

2. **数据要求**：
   - 展厅图片：`data/R.jpg` 或其他图片
   - 拓扑数据：`skills/topology/assets/TH.csv` 或 `OS.xls`

3. **输出位置**：所有结果保存在 `data/outputs/` 目录下

## 下一步建议

1. **收集真实眼动数据**：替换模拟数据，使用真实的眼动追踪数据
2. **运行完整实验**：在多个用户数据上运行对比实验和消融实验
3. **生成论文图表**：使用可视化结果生成 IROS 论文所需的高质量图表
4. **视频演示**：录制系统运行过程的演示视频

## Git 提交建议

```bash
git add -A
git commit -m "feat: 实现完整的 VLM + 眼动可视化系统，添加对比实验和演示流程

主要更新：
- 修复热力图和轨迹图可视化问题
- 添加 region_id 属性支持数据共享
- 创建统一可视化脚本 (unified_visualization.py)
- 实现对比实验和消融实验框架
- 添加完整演示流程和故事模式

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
