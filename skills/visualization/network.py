"""
Network Visualizer - 网络拓扑可视化器

This module provides network topology visualization for the exhibition hall
graph structure.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Optional
import os


class NetworkVisualizer:
    """
    网络可视化器

    This class creates network graph visualizations showing the topology
    and connectivity of the exhibition hall.
    """

    def __init__(self, topology_engine):
        """
        初始化网络可视化器

        Args:
            topology_engine: TopologyEngine实例
        """
        self.topology = topology_engine
        self.graph = topology_engine.graph

    def plot_network_graph(
        self,
        output_path: str = None,
        show: bool = False,
        layout: str = 'spring'
    ) -> plt.Figure:
        """
        绘制完整的网络拓扑图

        Args:
            output_path: 保存路径 (可选)
            show: 是否显示图片 (默认False)
            layout: 布局算法 ('spring', 'circular', 'kamada_kawai', 'spectral')

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(16, 12))

        # 选择布局
        pos = self._get_layout(layout)

        # 绘制边（根据关系类型着色）
        edge_colors = []
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', 'next')
            if relation == 'next':
                edge_colors.append('gray')
            elif relation == 'visual':
                edge_colors.append('blue')
            else:
                edge_colors.append('green')

        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color=edge_colors,
            width=1,
            alpha=0.4,
            ax=ax
        )

        # 绘制节点（根据类型着色）
        node_colors = []
        for node in self.graph.nodes:
            node_type = self.graph.nodes[node].get('type', 'General')
            if node_type == 'Entrance':
                node_colors.append('#d73027')  # 红色
            elif node_type == 'Exit':
                node_colors.append('#1a9850')  # 绿色
            elif node_type == 'Info':
                node_colors.append('#fee08b')  # 黄色
            else:
                node_colors.append('#abd9e9')  # 蓝色

        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=600,
            alpha=0.8,
            edgecolors='gray',
            linewidths=1,
            ax=ax
        )

        # 节点标签
        labels = {
            node: f"{node}\n{self.graph.nodes[node].get('name', 'Unknown')[:10]}"
            for node in self.graph.nodes
        }
        nx.draw_networkx_labels(
            self.graph, pos,
            labels,
            font_size=7,
            ax=ax
        )

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d73027', label='Entrance'),
            Patch(facecolor='#1a9850', label='Exit'),
            Patch(facecolor='#fee08b', label='Info'),
            Patch(facecolor='#abd9e9', label='Exhibit'),
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=10
        )

        # 统计信息
        stats_text = (
            f"Nodes: {self.graph.number_of_nodes()}\n"
            f"Edges: {self.graph.number_of_edges()}\n"
            f"Connected: {nx.is_weakly_connected(self.graph)}"
        )
        ax.text(
            0.02, 0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.set_title(
            f'Exhibition Hall Network Topology ({self.topology.map_name})',
            fontsize=16,
            fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Network graph saved to: {output_path}")

        if show:
            plt.show()

        return fig

    def plot_subgraph(
        self,
        center_node: str,
        radius: int = 2,
        output_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        绘制以某节点为中心的子图

        Args:
            center_node: 中心节点ID
            radius: 半径（跳数）
            output_path: 保存路径 (可选)
            show: 是否显示图片 (默认False)

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # 提取子图
        subgraph = nx.ego_graph(
            self.graph,
            center_node,
            radius=radius,
            undirected=False
        )

        # 布局
        pos = nx.spring_layout(subgraph, k=0.5, seed=42)

        # 绘制
        nx.draw_networkx_edges(
            subgraph, pos,
            alpha=0.3,
            ax=ax
        )

        # 中心节点特殊颜色
        node_colors = [
            '#d73027' if node == center_node else '#abd9e9'
            for node in subgraph.nodes
        ]

        node_sizes = [
            1000 if node == center_node else 600
            for node in subgraph.nodes
        ]

        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=ax
        )

        labels = {
            node: f"{node}\n{subgraph.nodes[node].get('name', 'Unknown')[:10]}"
            for node in subgraph.nodes
        }
        nx.draw_networkx_labels(
            subgraph, pos,
            labels,
            font_size=8,
            ax=ax
        )

        ax.set_title(
            f'Subgraph around {center_node} (radius={radius})',
            fontsize=14,
            fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Subgraph saved to: {output_path}")

        if show:
            plt.show()

        return fig

    def plot_shortest_path(
        self,
        start_node: str,
        end_node: str,
        output_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        绘制两节点间的最短路径

        Args:
            start_node: 起始节点ID
            end_node: 结束节点ID
            output_path: 保存路径 (可选)
            show: 是否显示图片 (默认False)

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        try:
            # 计算最短路径
            path = nx.shortest_path(self.graph, start_node, end_node)
            path_edges = list(zip(path, path[1:]))

            pos = nx.spring_layout(self.graph, k=0.5, seed=42)

            # 绘制所有边（浅色）
            nx.draw_networkx_edges(
                self.graph, pos,
                alpha=0.1,
                ax=ax
            )

            # 绘制所有节点（浅色）
            nx.draw_networkx_nodes(
                self.graph, pos,
                node_size=300,
                node_color='lightgray',
                alpha=0.5,
                ax=ax
            )

            # 高亮最短路径
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=path_edges,
                edge_color='red',
                width=3,
                ax=ax
            )

            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=path,
                node_size=700,
                node_color='red',
                alpha=0.8,
                ax=ax
            )

            # 标签
            labels = {node: node for node in path}
            nx.draw_networkx_labels(
                self.graph, pos,
                labels,
                font_size=10,
                font_weight='bold',
                ax=ax
            )

            ax.set_title(
                f'Shortest Path: {start_node} → {end_node} ({len(path)-1} steps)',
                fontsize=14,
                fontweight='bold'
            )

        except nx.NetworkXNoPath:
            ax.text(
                0.5, 0.5,
                f'No path found between {start_node} and {end_node}',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=14
            )
            ax.set_title('Shortest Path - Not Found', fontsize=14)

        ax.axis('off')
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Shortest path saved to: {output_path}")

        if show:
            plt.show()

        return fig

    def _get_layout(self, layout_name: str) -> Dict:
        """获取布局算法"""
        if layout_name == 'spring':
            return nx.spring_layout(self.graph, k=0.5, seed=42)
        elif layout_name == 'circular':
            return nx.circular_layout(self.graph)
        elif layout_name == 'kamada_kawai':
            return nx.kamada_kawai_layout(self.graph)
        elif layout_name == 'spectral':
            return nx.spectral_layout(self.graph)
        else:
            return nx.spring_layout(self.graph, k=0.5, seed=42)
