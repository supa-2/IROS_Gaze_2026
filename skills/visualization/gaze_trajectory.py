# -*- coding: utf-8 -*-
"""
眼动轨迹可视化器

显示观众在展厅中的观看顺序和移动路径
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrajectoryPoint:
    """轨迹点数据"""
    x: float
    y: float
    duration: float
    sequence_number: int
    region_id: str
    timestamp: datetime


class GazeTrajectoryVisualizer:
    """
    眼动轨迹可视化器

    功能：
    - 绘制凝视点（圆点，大小反映时长）
    - 绘制移动路径（连线）
    - 标注观看顺序（数字）
    """

    def __init__(self, image_path: str):
        """
        初始化轨迹可视化器

        Args:
            image_path: 原始图片路径
        """
        self.image_path = image_path
        self.original_image = Image.open(image_path).convert('RGB')
        self.image_width, self.image_height = self.original_image.size

        # 轨迹点列表（按时间顺序）
        self.trajectory_points: List[TrajectoryPoint] = []

        # 区域信息
        self.regions: Dict[str, Dict] = {}

    def add_region(self, region_id: str, label: str, bbox: List[int]):
        """
        添加区域信息

        Args:
            region_id: 区域ID
            label: 区域名称
            bbox: [x1, y1, x2, y2]
        """
        self.regions[region_id] = {
            'label': label,
            'bbox': bbox
        }

    def add_trajectory_point(
        self,
        x: float,
        y: float,
        duration: float,
        region_id: str,
        sequence_number: Optional[int] = None
    ):
        """
        添加轨迹点

        Args:
            x: x 坐标
            y: y 坐标
            duration: 凝视时长（秒）
            region_id: 所属区域ID
            sequence_number: 序号（可选，默认自动递增）
        """
        if sequence_number is None:
            sequence_number = len(self.trajectory_points) + 1

        point = TrajectoryPoint(
            x=x,
            y=y,
            duration=duration,
            sequence_number=sequence_number,
            region_id=region_id,
            timestamp=datetime.now()
        )
        self.trajectory_points.append(point)

    def add_gaze_records_from_visualizer(self, visualizer):
        """
        从热力图可视化器导入眼动记录
        """
        for i, fixation in enumerate(visualizer.fixations):
            self.add_trajectory_point(
                x=fixation.x,
                y=fixation.y,
                duration=fixation.duration,
                region_id=fixation.region_id if hasattr(fixation, 'region_id') else 'unknown',
                sequence_number=i + 1
            )

    def visualize(
        self,
        output_path: str = None,
        show: bool = False,
        style: str = 'default',
        show_numbers: bool = True,
        show_region_boxes: bool = True,
        line_width: float = 2.0,
        point_size_min: float = 50,
        point_size_max: float = 500,
        cmap: str = 'YlOrRd'
    ) -> str:
        """
        绘制眼动轨迹图

        Args:
            output_path: 输出路径
            show: 是否显示
            style: 风格 'default', 'minimal', 'detailed'
            show_numbers: 是否显示序号
            show_region_boxes: 是否显示区域边界框
            line_width: 连线宽度
            point_size_min: 最小点大小
            point_size_max: 最大点大小
            cmap: 颜色映射（用于时长）
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/outputs/gaze_trajectory_{timestamp}.png"

        import os
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 10))

        # 显示原图
        img_array = np.array(self.original_image)
        ax.imshow(img_array)

        # 绘制区域边界框
        if show_region_boxes:
            for region_id, region_info in self.regions.items():
                bbox = region_info['bbox']
                label = region_info['label']
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    linewidth=2,
                    edgecolor='white',
                    facecolor='none',
                    alpha=0.7
                )
                ax.add_patch(rect)

                # 添加区域标签
                ax.text(
                    bbox[0], bbox[1] - 15,
                    label,
                    color='white',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5)
                )

        if not self.trajectory_points:
            ax.set_title("Gaze Trajectory - No Data", fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"[Saved] {output_path}")
            if not show:
                plt.close()
            return output_path

        # 获取时长范围用于点大小
        durations = [p.duration for p in self.trajectory_points]
        duration_min = min(durations) if durations else 1
        duration_max = max(durations) if durations else 1

        # 获取颜色映射
        colormap = plt.get_cmap(cmap)

        # 按序号排序
        sorted_points = sorted(self.trajectory_points, key=lambda p: p.sequence_number)

        # 绘制连接线
        if len(sorted_points) > 1:
            x_coords = [p.x for p in sorted_points]
            y_coords = [p.y for p in sorted_points]

            # 绘制连线
            ax.plot(
                x_coords, y_coords,
                color='white',
                linewidth=line_width,
                alpha=0.8,
                linestyle='-',
                zorder=1,
                marker='o',
                markersize=4,
                markerfacecolor='white',
                markeredgecolor='black',
                markeredgewidth=1
            )

        # 绘制每个凝视点
        for point in sorted_points:
            # 根据时长计算点大小
            if duration_max > duration_min:
                normalized_duration = (point.duration - duration_min) / (duration_max - duration_min)
            else:
                normalized_duration = 0.5

            point_size = point_size_min + normalized_duration * (point_size_max - point_size_min)

            # 根据时长获取颜色
            color = colormap(normalized_duration)

            # 绘制点
            ax.scatter(
                point.x, point.y,
                s=point_size,
                c=[color],
                alpha=0.7,
                edgecolors='white',
                linewidths=2,
                zorder=2
            )

            # 绘制序号
            if show_numbers:
                ax.text(
                    point.x, point.y,
                    str(point.sequence_number),
                    color='black',
                    fontsize=10,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', alpha=0.9),
                    zorder=3
                )

        # 添加标题和图例
        title = f"Gaze Trajectory ({len(sorted_points)} fixations)"
        ax.set_title(title, fontsize=14, fontweight='bold', color='white')
        ax.axis('off')

        # 添加颜色条（时长）
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Fixation Duration (s)', rotation=270, labelpad=20, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')

        # 设置背景色（使文字更清晰）
        fig.patch.set_facecolor('#333333')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.patch.get_facecolor())
        print(f"[Saved] {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def visualize_with_heatmap(
        self,
        heatmap_data: np.ndarray,
        output_path: str = None,
        show: bool = False,
        trajectory_alpha: float = 1.0,
        heatmap_alpha: float = 0.5,
        heatmap_cmap: str = 'jet'
    ) -> str:
        """
        叠加轨迹和热力图

        Args:
            heatmap_data: 热力图数据
            output_path: 输出路径
            show: 是否显示
            trajectory_alpha: 轨迹透明度
            heatmap_alpha: 热力图透明度
            heatmap_cmap: 热力图配色
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/outputs/gaze_trajectory_heatmap_{timestamp}.png"

        import os
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 10))

        # 显示原图
        img_array = np.array(self.original_image)
        ax.imshow(img_array)

        # 叠加热力图
        im = ax.imshow(
            heatmap_data,
            cmap=heatmap_cmap,
            alpha=heatmap_alpha,
            vmin=0,
            vmax=1
        )

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Heat Intensity', rotation=270, labelpad=20)

        # 绘制区域边界框
        for region_id, region_info in self.regions.items():
            bbox = region_info['bbox']
            label = region_info['label']
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=2,
                edgecolor='yellow',
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1] - 15, label, color='yellow',
                    fontsize=10, fontweight='bold')

        # 绘制轨迹
        if self.trajectory_points:
            sorted_points = sorted(self.trajectory_points, key=lambda p: p.sequence_number)

            # 连线
            if len(sorted_points) > 1:
                x_coords = [p.x for p in sorted_points]
                y_coords = [p.y for p in sorted_points]
                ax.plot(x_coords, y_coords, color='white', linewidth=2.5,
                        alpha=trajectory_alpha, marker='o', markersize=6,
                        markerfacecolor='white', markeredgecolor='black', zorder=3)

            # 点和序号
            for point in sorted_points:
                # 点
                ax.scatter(point.x, point.y, s=200, c='white',
                          edgecolors='black', linewidths=2, alpha=trajectory_alpha, zorder=4)

                # 序号
                ax.text(point.x, point.y, str(point.sequence_number),
                        color='black', fontsize=11, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', alpha=0.95),
                        zorder=5)

        ax.set_title("Gaze Trajectory + Heatmap Overlay", fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def get_statistics(self) -> Dict:
        """获取轨迹统计信息"""
        if not self.trajectory_points:
            return {}

        # 按区域统计
        region_stats = {}
        for point in self.trajectory_points:
            rid = point.region_id
            if rid not in region_stats:
                region_stats[rid] = {
                    'count': 0,
                    'total_duration': 0,
                    'first_visit': None,
                    'last_visit': None
                }
            region_stats[rid]['count'] += 1
            region_stats[rid]['total_duration'] += point.duration

            if region_stats[rid]['first_visit'] is None:
                region_stats[rid]['first_visit'] = point.sequence_number
            region_stats[rid]['last_visit'] = point.sequence_number

        # 计算转换次数
        transitions = 0
        sorted_points = sorted(self.trajectory_points, key=lambda p: p.sequence_number)
        for i in range(len(sorted_points) - 1):
            if sorted_points[i].region_id != sorted_points[i+1].region_id:
                transitions += 1

        return {
            'total_fixations': len(self.trajectory_points),
            'total_duration': sum(p.duration for p in self.trajectory_points),
            'unique_regions': len(set(p.region_id for p in self.trajectory_points)),
            'region_transitions': transitions,
            'region_stats': region_stats
        }
