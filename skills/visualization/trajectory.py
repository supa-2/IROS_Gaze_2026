"""
Trajectory Visualizer - 轨迹可视化器

This module provides trajectory visualization for historical and predicted
gaze paths through the exhibition hall.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict
import os
from skills.memory.manager import GazeRecord


class TrajectoryVisualizer:
    """
    眼动轨迹可视化器

    This class creates trajectory visualizations showing the sequence
    of exhibits viewed by users, both for historical and predicted paths.
    """

    def __init__(self, topology_engine):
        """
        初始化轨迹可视化器

        Args:
            topology_engine: TopologyEngine实例
        """
        self.topology = topology_engine
        self.graph = topology_engine.graph

    def plot_gaze_trajectory(
        self,
        gaze_sequence: List[GazeRecord],
        output_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        绘制历史眼动轨迹

        Args:
            gaze_sequence: 凝视序列列表
            output_path: 保存路径 (可选)
            show: 是否显示图片 (默认False)

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # 计算布局
        pos = nx.spring_layout(self.graph, k=0.5, seed=42)

        # 绘制底图（所有节点和边）
        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.1,
            edge_color='gray',
            width=1,
            ax=ax
        )
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=200,
            node_color='lightgray',
            alpha=0.5,
            ax=ax
        )

        # 提取路径
        path_ids = [g.exhibit_id for g in gaze_sequence]
        path_edges = list(zip(path_ids, path_ids[1:]))

        # 绘制轨迹边（红色加粗）
        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=path_edges,
            edge_color='red',
            width=3,
            alpha=0.7,
            ax=ax
        )

        # 绘制轨迹节点（红色）
        nx.draw_networkx_nodes(
            self.graph, pos,
            nodelist=path_ids,
            node_size=700,
            node_color='#ff6b6b',
            alpha=0.9,
            edgecolors='darkred',
            linewidths=2,
            ax=ax
        )

        # 添加序号和注意力等级标签
        labels = {}
        for i, gaze in enumerate(gaze_sequence):
            node_id = gaze.exhibit_id
            level = gaze.attention_level
            labels[node_id] = f"{i+1}\n[{level}]"

        nx.draw_networkx_labels(
            self.graph, pos,
            labels,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        # 添加图例
        ax.text(
            0.02, 0.98,
            f'Trajectory: {len(path_ids)} steps\n'
            f'Total time: {sum(g.estimated_duration for g in gaze_sequence)}s',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        ax.set_title(
            f'Historical Gaze Trajectory ({len(path_ids)} exhibits)',
            fontsize=16,
            fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory saved to: {output_path}")

        if show:
            plt.show()

        return fig

    def plot_predicted_trajectory(
        self,
        predictions: List[Dict],
        output_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        绘制预测轨迹（虚线表示）

        Args:
            predictions: 预测序列列表
            output_path: 保存路径 (可选)
            show: 是否显示图片 (默认False)

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        pos = nx.spring_layout(self.graph, k=0.5, seed=42)

        # 底图
        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.1,
            edge_color='gray',
            ax=ax
        )
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=200,
            node_color='lightgray',
            alpha=0.5,
            ax=ax
        )

        # 预测路径
        pred_ids = [p['prediction_id'] for p in predictions]
        pred_edges = list(zip(pred_ids, pred_ids[1:]))

        # 轨迹边（蓝色虚线）
        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=pred_edges,
            edge_color='blue',
            width=2,
            style='dashed',
            alpha=0.6,
            ax=ax
        )

        # 轨迹节点（根据注意力等级着色）
        node_colors = [
            {'A': '#d73027', 'B': '#fc8d59', 'C': '#fee08b',
             'D': '#d9ef8b', 'E': '#1a9850'}.get(
                p.get('attention_level', 'C'), 'gray'
            )
            for p in predictions
        ]

        nx.draw_networkx_nodes(
            self.graph, pos,
            nodelist=pred_ids,
            node_size=700,
            node_color=node_colors,
            alpha=0.9,
            edgecolors='navy',
            linewidths=2,
            ax=ax
        )

        # 标签（步数+注意力等级+时间）
        labels = {}
        for i, pred in enumerate(predictions):
            pred_id = pred['prediction_id']
            level = pred.get('attention_level', '?')
            duration = pred.get('estimated_duration', 0)
            conf = pred.get('confidence', 0.0)
            labels[pred_id] = f"{i+1}\n[{level}]\n{duration}s\n({conf:.2f})"

        nx.draw_networkx_labels(
            self.graph, pos,
            labels,
            font_size=9,
            font_weight='bold',
            ax=ax
        )

        # 图例
        total_duration = sum(p.get('estimated_duration', 0) for p in predictions)
        avg_confidence = sum(p.get('confidence', 0) for p in predictions) / len(predictions) if predictions else 0

        ax.text(
            0.02, 0.98,
            f'Predicted Path: {len(pred_ids)} steps\n'
            f'Est. Duration: {total_duration}s ({total_duration/60:.1f}min)\n'
            f'Avg Confidence: {avg_confidence:.2f}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
        )

        ax.set_title(
            f'Predicted Gaze Trajectory ({len(pred_ids)} steps)',
            fontsize=16,
            fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Predicted trajectory saved to: {output_path}")

        if show:
            plt.show()

        return fig

    def plot_comparison_trajectories(
        self,
        historical: List[GazeRecord],
        predicted: List[Dict],
        output_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        并排对比历史轨迹和预测轨迹

        Args:
            historical: 历史轨迹
            predicted: 预测轨迹
            output_path: 保存路径 (可选)
            show: 是否显示图片 (默认False)

        Returns:
            matplotlib Figure对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        pos = nx.spring_layout(self.graph, k=0.5, seed=42)

        # 左图：历史轨迹
        self._draw_trajectory_on_axis(
            ax1, pos, historical, None,
            title='Historical Trajectory',
            color='red'
        )

        # 右图：预测轨迹
        self._draw_predicted_trajectory_on_axis(
            ax2, pos, predicted,
            title='Predicted Trajectory',
            color='blue'
        )

        plt.suptitle(
            'Historical vs Predicted Trajectories',
            fontsize=16,
            fontweight='bold'
        )

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to: {output_path}")

        if show:
            plt.show()

        return fig

    def _draw_trajectory_on_axis(self, ax, pos, gaze_sequence, title, color):
        """在指定轴上绘制轨迹"""
        # 底图
        nx.draw_networkx_edges(self.graph, pos, alpha=0.1, ax=ax)
        nx.draw_networkx_nodes(self.graph, pos, node_size=150, node_color='lightgray', ax=ax)

        # 轨迹
        path_ids = [g.exhibit_id for g in gaze_sequence]
        path_edges = list(zip(path_ids, path_ids[1:]))

        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=path_edges,
            edge_color=color,
            width=2,
            ax=ax
        )

        nx.draw_networkx_nodes(
            self.graph, pos,
            nodelist=path_ids,
            node_size=500,
            node_color=color,
            alpha=0.8,
            ax=ax
        )

        labels = {node: str(i+1) for i, node in enumerate(path_ids)}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=9, ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    def _draw_predicted_trajectory_on_axis(self, ax, pos, predictions, title, color):
        """在指定轴上绘制预测轨迹"""
        # 底图
        nx.draw_networkx_edges(self.graph, pos, alpha=0.1, ax=ax)
        nx.draw_networkx_nodes(self.graph, pos, node_size=150, node_color='lightgray', ax=ax)

        # 轨迹
        pred_ids = [p['prediction_id'] for p in predictions]
        pred_edges = list(zip(pred_ids, pred_ids[1:]))

        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=pred_edges,
            edge_color=color,
            width=2,
            style='dashed',
            ax=ax
        )

        nx.draw_networkx_nodes(
            self.graph, pos,
            nodelist=pred_ids,
            node_size=500,
            node_color=color,
            alpha=0.8,
            ax=ax
        )

        labels = {node: str(i+1) for i, node in enumerate(pred_ids)}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=9, ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
