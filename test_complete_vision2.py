"""
完整测试：VLM + 眼动数据 → 原图热力图
"""

import sys
import os

# 设置 UTF-8 输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv

# 导入 agent
sys.path.append(os.path.join(os.path.dirname(__file__)))
from agent import EyeLLMAgent

def main():
    print("=" * 70)
    print("  Eye-LLM 完整视觉系统测试")
    print("  VLM + 眼动数据 → 原图热力图")
    print("=" * 70)

    # 初始化 agent (使用新架构)
    print("\n[1] 初始化 Agent...")
    agent = EyeLLMAgent(map_name='TH', use_new_architecture=True)
    print("    Agent 已就绪")

    # 测试图片路径
    test_image = "data/test_museum.jpg"

    if not os.path.exists(test_image):
        print(f"\n    测试图片不存在: {test_image}")
        print("    请先运行 test_vision.py 生成测试图片")
        return

    # 模拟 VLM 识别的展品区域
    print("\n[2] 模拟 VLM 识别的展品区域...")
    mock_vlm_regions = [
        {"id": "zone_a", "label": "青铜器区", "type": "Exhibit", "bbox": [100, 150, 300, 350]},
        {"id": "zone_b", "label": "陶瓷展品区", "type": "Exhibit", "bbox": [400, 100, 550, 300]},
        {"id": "zone_c", "label": "说明牌区", "type": "Label", "bbox": [500, 350, 650, 450]},
        {"id": "zone_d", "label": "出口通道", "type": "Passage", "bbox": [600, 400, 800, 550]},
    ]

    # 添加 VLM 识别的区域到像素热力图可视化器
    print("    添加区域到像素热力图可视化器...")
    for region in mock_vlm_regions:
        agent.pixel_heatmap_viz.add_vlm_regions([region])

    # 模拟眼动数据：用户在不同区域的观看时长
    print("\n[3] 添加模拟眼动数据...")
    gaze_data = [
        # 在青铜器区看了 45秒
        {"zone_id": "zone_a", "duration": 45.0},
        {"zone_id": "zone_a", "duration": 30.0},
        # 在陶瓷展品区看了 60秒
        {"zone_id": "zone_b", "duration": 60.0},
        # 在说明牌区看了 20秒
        {"zone_id": "zone_c", "duration": 20.0},
        # 在出口通道看了 15秒
        {"zone_id": "zone_d", "duration": 15.0},
    ]

    for data in gaze_data:
        agent.pixel_heatmap_viz.add_gaze_record(
            region_id=data["zone_id"],
            duration=data["duration"]
        )

    # 获取统计
    print("\n[4] 区域统计:")
    stats = agent.pixel_heatmap_viz.get_region_statistics()
    for region_id, stat in stats.items():
        print(f"    {stat['label']}:")
        print(f"      观看次数: {stat['view_count']}")
        print(f"      总时长: {stat['total_duration']:.1f}s")
        print(f"      平均时长: {stat['avg_duration']:.1f}s")

    # 生成并保存叠加热力图
    print("\n[5] 生成像素级热力图...")
    output_path = agent.pixel_heatmap_viz.visualize_overlay(
        output_path="data/outputs/pixel_heatmap_complete.png"
    )

    print(f"\n    热力图已保存到: {output_path}")

    print("\n" + "=" * 70)
    print("  测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
