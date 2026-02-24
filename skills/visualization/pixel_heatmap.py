"""
眼动热力图可视化器

基于凝视点（fixation points）+ 高斯模糊生成平滑热力图叠加在原图上

标准眼动热力图特征：
- 每个凝视点产生高斯模糊的"光晕"
- 颜色映射：红(最热) → 橙 → 黄 → 绿 → 蓝(最冷)
- 半透明叠加在原图上
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FixationPoint:
    """单个凝视点数据"""
    x: int           # 像素 x 坐标
    y: int           # 像素 y 坐标
    duration: float  # 凝视时长(秒)
    timestamp: datetime
    region_id: str = ''  # 所属区域ID（用于轨迹可视化）

    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat(),
            'region_id': self.region_id
        }


@dataclass
class GazeRegion:
    """VLM 识别的展品区域"""
    id: str
    label: str
    type: str              # Exhibit, Label, Passage
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def contains_point(self, x: int, y: int) -> bool:
        """检查点是否在区域内"""
        x1, y1, x2, y2 = self.bbox
        return x1 <= x < x2 and y1 <= y < y2


class GazeHeatmapVisualizer:
    """
    眼动热力图可视化器

    使用高斯模糊生成平滑的热力图叠加在原图上
    """

    def __init__(self, image_path: str, sigma: float = 50.0):
        """
        初始化可视化器

        Args:
            image_path: 原始图片路径
            sigma: 高斯模糊标准差，控制光晕大小(默认50像素)
        """
        self.image_path = image_path
        self.sigma = sigma

        # 加载原图
        self.original_image = Image.open(image_path).convert('RGB')
        self.image_width, self.image_height = self.original_image.size

        # VLM 识别的区域
        self.regions: Dict[str, GazeRegion] = {}

        # 凝视点数据
        self.fixations: List[FixationPoint] = []

        # 生成的热力图
        self.heatmap_data: Optional[np.ndarray] = None

    def add_vlm_regions(self, regions: List[Dict]):
        """
        添加 VLM 识别的区域

        Args:
            regions: [{"id": str, "label": str, "type": str, "bbox": [x1,y1,x2,y2]}, ...]
        """
        for region_dict in regions:
            region = GazeRegion(
                id=region_dict['id'],
                label=region_dict['label'],
                type=region_dict['type'],
                bbox=tuple(region_dict['bbox'])
            )
            self.regions[region.id] = region

    def add_gaze_record(
        self,
        region_id: str,
        duration: float,
        x: Optional[int] = None,
        y: Optional[int] = None
    ):
        """
        添加眼动观测记录

        Args:
            region_id: 展品区域ID
            duration: 凝视时长(秒)
            x: 凝视点x坐标(可选，默认使用区域中心)
            y: 凝视点y坐标(可选，默认使用区域中心)
        """
        region = self.regions.get(region_id)
        if not region:
            print(f"[Warning] Region {region_id} not found")
            return

        # 如果没有指定坐标，使用区域中心
        if x is None or y is None:
            x, y = region.center

        fixation = FixationPoint(
            x=int(x),
            y=int(y),
            duration=duration,
            timestamp=datetime.now(),
            region_id=region_id
        )
        self.fixations.append(fixation)
        print(f"[GAZE] {region.label}: ({x},{y}) - {duration}s")

    def calculate_heatmap(self, decay_factor: float = 0.95) -> np.ndarray:
        """
        计算热力图

        为每个凝视点创建高斯模糊的"光晕"，权重由时长决定

        Args:
            decay_factor: 时间衰减因子(默认0.95)
        """
        # 创建空白热力图
        heatmap = np.zeros((self.image_height, self.image_width), dtype=np.float32)

        if not self.fixations:
            print("[Warning] No fixation data available")
            return heatmap

        # 计算当前时间用于衰减
        current_time = datetime.now()

        # 为每个凝视点添加热力值
        for fixation in self.fixations:
            # 时间衰减
            time_diff = (current_time - fixation.timestamp).total_seconds()
            decay = decay_factor ** (time_diff / 600.0)  # 10分钟半衰期
            weighted_duration = fixation.duration * decay

            # 检查坐标是否在范围内
            if 0 <= fixation.y < self.image_height and 0 <= fixation.x < self.image_width:
                heatmap[fixation.y, fixation.x] += weighted_duration

        # 应用高斯模糊生成平滑热力分布
        if heatmap.max() > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.sigma)

        # 归一化到 0-1
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        self.heatmap_data = heatmap
        return heatmap

    def get_region_statistics(self) -> Dict:
        """获取各区域的眼动统计"""
        stats = {}

        for region_id, region in self.regions.items():
            # 找到该区域内的所有凝视点
            region_fixations = [
                f for f in self.fixations
                if region.contains_point(f.x, f.y)
            ]

            if region_fixations:
                total_duration = sum(f.duration for f in region_fixations)
                avg_duration = total_duration / len(region_fixations)

                stats[region_id] = {
                    'label': region.label,
                    'type': region.type,
                    'fixation_count': len(region_fixations),
                    'total_duration': total_duration,
                    'avg_duration': avg_duration,
                    'bbox': region.bbox,
                    'center': region.center
                }

        return stats

    def visualize_overlay(
        self,
        output_path: str = None,
        show: bool = False,
        alpha: float = 0.6,
        cmap: str = 'jet',
        vmin: float = 0.0,
        vmax: float = 1.0
    ) -> str:
        """
        生成热力图叠加可视化

        Args:
            output_path: 输出路径
            show: 是否显示
            alpha: 热力图透明度(0-1)
            cmap: 颜色映射(jet, hot, coolwarm等)
            vmin, vmax: 热力值范围
        """
        if self.heatmap_data is None:
            self.calculate_heatmap()

        # 转换原图为numpy数组
        img_array = np.array(self.original_image)

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))

        # 显示原图
        ax.imshow(img_array)

        # 叠加热力图
        im = ax.imshow(
            self.heatmap_data,
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear'
        )

        ax.set_title("Eye Tracking Heatmap Overlay", fontsize=14, fontweight='bold')
        ax.axis('off')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Gaze Intensity (normalized)', rotation=270, labelpad=20)

        plt.tight_layout()

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/outputs/gaze_heatmap_{timestamp}.png"

        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def visualize_side_by_side(
        self,
        output_path: str = None,
        show: bool = False
    ) -> str:
        """
        生成并排对比图：原图 | 热力图 | 叠加图
        """
        if self.heatmap_data is None:
            self.calculate_heatmap()

        img_array = np.array(self.original_image)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 左：原图
        axes[0].imshow(img_array)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 中：纯热力图
        im = axes[1].imshow(
            self.heatmap_data,
            cmap='jet',
            vmin=0,
            vmax=1
        )
        axes[1].set_title("Heatmap Only", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # 右：叠加
        axes[2].imshow(img_array)
        axes[2].imshow(
            self.heatmap_data,
            cmap='jet',
            alpha=0.5,
            vmin=0,
            vmax=1
        )
        axes[2].set_title("Overlay", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/outputs/gaze_comparison_{timestamp}.png"

        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path
