"""
动态拓扑引擎 - 从 VLM 识别结果动态构建拓扑图

用于处理从新图片中识别出的展品，无需预定义的拓扑文件
"""

import networkx as nx
from typing import List, Dict, Optional


class DynamicTopologyEngine:
    """
    动态拓扑引擎

    从 VLM 识别结果构建拓扑图，支持动态添加展品和关系
    """

    def __init__(self, map_name: str = "DYNAMIC"):
        self.graph = nx.DiGraph()
        self.map_name = map_name

    def load_from_vlm_result(self, vlm_result: Dict):
        """
        从 VLM 识别结果加载拓扑

        Args:
            vlm_result: VLM 识别结果，包含 exhibits 和 relationships
        """
        exhibits = vlm_result.get('exhibits', [])
        relationships = vlm_result.get('relationships', [])

        # 添加节点
        for exhibit in exhibits:
            self.graph.add_node(
                exhibit['id'],
                name=exhibit.get('name', 'Unknown'),
                type=exhibit.get('type', 'General'),
                features=exhibit.get('visual_features', ''),
                bbox=exhibit.get('bbox', []),
                center=exhibit.get('center', []),
                description=exhibit.get('description', ''),
                attention_level=exhibit.get('attention_level', 'C')
            )

        # 添加边
        for rel in relationships:
            self.graph.add_edge(
                rel['source'],
                rel['target'],
                relation=rel.get('relation', 'next_to')
            )

        # 如果没有显式关系，根据空间位置自动推断
        if not relationships and len(exhibits) > 1:
            self._infer_spatial_relations(exhibits)

        print(f"[Topology] 加载了 {len(exhibits)} 个节点, {self.graph.number_of_edges()} 条边")

    def _infer_spatial_relations(self, exhibits: List[Dict]):
        """根据空间位置推断关系"""
        # 按中心点坐标排序（从左到右，从上到下）
        sorted_exhibits = sorted(exhibits, key=lambda e: (e['center'][1], e['center'][0]))

        for i in range(len(sorted_exhibits) - 1):
            current = sorted_exhibits[i]
            next_ex = sorted_exhibits[i + 1]

            # 计算距离
            dx = next_ex['center'][0] - current['center'][0]
            dy = next_ex['center'][1] - current['center'][1]
            distance = (dx ** 2 + dy ** 2) ** 0.5

            # 添加连接关系
            if distance < 500:  # 像素距离阈值
                self.graph.add_edge(current['id'], next_ex['id'], relation='next_to')

    def query_node(self, node_id: str) -> Dict:
        """
        查询节点信息

        Args:
            node_id: 展品ID

        Returns:
            节点信息字典
        """
        if node_id not in self.graph:
            return {"error": f"Node {node_id} not found."}

        node_info = self.graph.nodes[node_id]

        # 获取直接邻居
        direct_connected = []
        for n in self.graph.successors(node_id):
            direct_connected.append({
                "id": n,
                "relation": self.graph[node_id][n]['relation']
            })

        # 获取前序和后续路径
        prev_chain = self._trace_path(node_id, 'predecessors', 3)
        next_chain = self._trace_path(node_id, 'successors', 3)

        return {
            "map": self.map_name,
            "id": node_id,
            "info": node_info,
            "context": {
                "previous_path": prev_chain,
                "next_path": next_chain,
                "direct_choices": direct_connected
            }
        }

    def _trace_path(self, start_node: str, direction: str = 'successors', depth: int = 3) -> List[Dict]:
        """追踪路径"""
        path = []
        curr = start_node
        visited = {start_node}

        for _ in range(depth):
            if direction == 'successors':
                neighbors = list(self.graph.successors(curr))
            else:
                neighbors = list(self.graph.predecessors(curr))

            if not neighbors:
                break

            # 选择第一个未访问的邻居
            next_node = None
            for n in neighbors:
                if n not in visited:
                    next_node = n
                    break

            if next_node is None:
                break

            info = self.graph.nodes[next_node]
            path.append({
                "id": next_node,
                "name": info.get('name'),
                "dist": len(path) + 1
            })

            visited.add(next_node)
            curr = next_node

        return path

    def get_all_nodes(self) -> List[Dict]:
        """获取所有节点信息"""
        nodes = []
        for node_id in self.graph.nodes():
            info = self.graph.nodes[node_id]
            nodes.append({
                "id": node_id,
                "name": info.get('name'),
                "type": info.get('type'),
                "features": info.get('features')
            })
        return nodes

    def get_reachable_exhibits(self, exhibit_id: str) -> List[Dict]:
        """获取从指定展品可达的所有展品"""
        if exhibit_id not in self.graph:
            return []

        reachable = []
        for neighbor in self.graph.successors(exhibit_id):
            info = self.graph.nodes[neighbor]
            reachable.append({
                "id": neighbor,
                "name": info.get('name'),
                "relation": self.graph[exhibit_id][neighbor]['relation']
            })
        return reachable

    def get_entry_point(self) -> Optional[str]:
        """获取入口点（第一个没有前驱的节点）"""
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                return node
        # 如果所有节点都有前驱，返回第一个节点
        if self.graph.number_of_nodes() > 0:
            return list(self.graph.nodes())[0]
        return None

    def get_recommended_path(self) -> List[str]:
        """获取推荐的观看路径"""
        # 使用拓扑排序获取合理的路径
        try:
            return list(nx.topological_sort(self.graph))
        except:
            # 如果有环，按节点添加顺序返回
            return list(self.graph.nodes())
