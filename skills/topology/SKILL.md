---
name: museum-topology
description: 博物馆空间拓扑与展项关系查询服务。用于获取展项的详细信息、物理邻居及语义关联，辅助预测用户意图。
---

# Museum Topology Skill

这是一个基于图数据的空间查询工具，用于帮助 Agent 理解博物馆内的空间结构和展项关系。

## Capabilities

### 1. 查询展项详情与邻居 (Query Node Context)
当需要了解某个具体展项（如 "OS-A01"）的类型、视觉特征或它连接了哪些其他展项时使用。

- **Command**: `python graph_engine.py --action query --id [EXHIBIT_ID]`
- **Output**: JSON 格式的展项信息，包含 `neighbors` 列表（含物理和语义关系）。

### 2. 预测推荐 (Predict Next)
当需要根据当前位置预测可能的下一个目标时使用。该命令会基于图的 `Flow` 和 `Visual` 权重计算推荐列表。

- **Command**: `python graph_engine.py --action predict --from [CURRENT_ID]`
- **Output**: 推荐的下一个展项 ID 列表及其置信度原因。

## Usage Rules
1. **ID 格式**：必须严格使用 CSV 中的 ID 格式（如 `OS-I-B05`），不要自己编造 ID。
2. **错误处理**：如果返回 "Node not found"，尝试搜索该名称的模糊匹配或询问用户具体位置。
3. **数据来源**：所有数据均源自 `assets/exhibits.csv`，这是项目的唯一真理来源 (Single Source of Truth)。