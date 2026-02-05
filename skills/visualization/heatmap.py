"""
Heatmap Visualizer - 热力图可视化器

This module provides heatmap visualization for visit frequency and dwell time
analysis using matplotlib and networkx.
"""

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Dict, List
import numpy as np
import os


class HeatmapVisualizer:
    """
    热力图可视化器

    This class creates heatmap visualizations showing visit frequency
    and dwell time distributions across the exhibition hall topology.
    """

    def __init__(self, topology_engine):
        """
        初始化热力图可视化器

        Args:
            topology_engine: TopologyEngine实例
        """
        self.topology = topology_engine
        self.graph = topology_engine.graph

        # 设置可视化风格
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False

    def plot_visit_heatmap(
        self,
        visit_counts: Dict[str, int],
        output_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        绘制访问频率热力图

        Args:
            visit_counts: {exhibit_id: count} 访问次数字典
            output_path: 保存路径 (可选)
            show: 是否显示图片 (默认False)

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # 计算布局
        pos = nx.spring_layout(self.graph, k=0.5, seed=42)

        # 准备颜色映射
        max_count = max(visit_counts.values()) if visit_counts else 1
        node_colors = [
            visit_counts.get(node, 0) / max_count
            for node in self.graph.nodes
        ]

        # 绘制边
        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.2,
            edge_color='gray',
            ax=ax
        )

        # 绘制节点
        nodes = nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            cmap='Reds',
            node_size=800,
            alpha=0.8,
            edgecolors='gray',
            linewidths=1,
            ax=ax
        )

        # 添加颜色条
        sm = plt.cm.ScalarMappable(
            cmap='Reds',
            norm=plt.Normalize(vmin=0, vmax=max_count)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Visit Frequency', shrink=0.8)
        cbar.ax.yaxis.label.set_size(12)

        # 添加节点标签
        labels = {
            node: f"{node}\n({visit_counts.get(node, 0)})"
            for node in self.graph.nodes
        }
        nx.draw_networkx_labels(
            self.graph, pos,
            labels,
            font_size=8,
            ax=ax
        )

        ax.set_title('Exhibition Hall Visit Frequency Heatmap', fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        # 保存
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {output_path}")

        # 显示
        if show:
            plt.show()

        return fig

    def plot_duration_heatmap(
        self,
        duration_data: Dict[str, int],
        output_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        绘制停留时间热力图

        Args:
            duration_data: {exhibit_id: total_seconds} 总停留时间字典
            output_path: 保存路径 (可选)
            show: 是否显示图片 (默认False)

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        pos = nx.spring_layout(self.graph, k=0.5, seed=42)

        # 时间映射
        max_duration = max(duration_data.values()) if duration_data else 1
        node_colors = [
            duration_data.get(node, 0) / max_duration
            for node in self.graph.nodes
        ]

        # 节点大小根据停留时间
        node_sizes = [
            300 + duration_data.get(node, 0) / 5
            for node in self.graph.nodes
        ]

        # 绘制
        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.2,
            edge_color='gray',
            ax=ax
        )

        nodes = nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            cmap='YlOrRd',  # 黄-橙-红
            node_size=node_sizes,
            alpha=0.8,
            edgecolors='gray',
            linewidths=1,
            ax=ax
        )

        # 颜色条
        sm = plt.cm.ScalarMappable(
            cmap='YlOrRd',
            norm=plt.Normalize(vmin=0, vmax=max_duration)
        )
        sm.set_array([])
        cbar = plt.colorbar(
            sm,
            ax=ax,
            label='Total Duration (seconds)',
            shrink=0.8
        )
        cbar.ax.yaxis.label.set_size(12)

        # 标签
        labels = {
            node: f"{node}\n{duration_data.get(node, 0)}s"
            for node in self.graph.nodes
        }
        nx.draw_networkx_labels(
            self.graph, pos,
            labels,
            font_size=7,
            ax=ax
        )

        ax.set_title('Exhibition Hall Dwell Time Heatmap', fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Duration heatmap saved to: {output_path}")

        if show:
            plt.show()

        return fig

    def plot_attention_distribution(
        self,
        attention_data: Dict[str, int],
        output_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        绘制注意力等级分布柱状图

        Args:
            attention_data: {level: count} 注意力等级统计
            output_path: 保存路径 (可选)
            show: 是否显示图片 (默认False)

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        levels = ['A', 'B', 'C', 'D', 'E']
        counts = [attention_data.get(level, 0) for level in levels]
        colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850']

        bars = ax.bar(levels, counts, color=colors, alpha=0.7, edgecolor='black')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=12
            )

        ax.set_xlabel('Attention Level', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Attention Level Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 设置x轴标签
        level_labels = ['A (120s)', 'B (60s)', 'C (30s)', 'D (15s)', 'E (5s)']
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(level_labels, rotation=0)

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Attention distribution saved to: {output_path}")

        if show:
            plt.show()

        return fig
