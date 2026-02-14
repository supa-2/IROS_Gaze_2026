# -*- coding: utf-8 -*-
"""
下载并配置本地 SAM 2 模型
"""

import os
import sys
import io

# UTF-8 output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 70)
print("  SAM 2 本地模型部署")
print("=" * 70)

# SAM 2 官方下载地址
SAM2_MODELS = {
    "tiny": "https://github.com/facebookresearch/segment-anything-2/main/releases/download/v2.1.0/sam2_hiera_tiny.pt",
    "small": "https://github.com/facebookresearch/segment-anything-2/main/releases/download/v2.1.0/sam2_hiera_small.pt",
    "base": "https://github.com/facebookresearch/segment-anything-2/main/releases/download/v2.1.0/sam2_hiera_base.pt",
}

# 选择版本（推荐 tiny 或 small）
MODEL_TYPE = "tiny"  # 可以改为 "small"

# 创建模型目录
MODELS_DIR = "models/sam2"
os.makedirs(MODELS_DIR, exist_ok=True)

# 模型文件路径
MODEL_PATH = os.path.join(MODELS_DIR, f"sam2_hiera_{MODEL_TYPE}.pt")

# 配置文件路径
CONFIG_PATH = os.path.join(MODELS_DIR, "sam2_config.yaml")

print(f"\n[1] 模型版本: {MODEL_TYPE}")
print(f"[2] 模型URL: {SAM2_MODELS[MODEL_TYPE]}")
print(f"[3] 保存目录: {MODELS_DIR}")
print(f"[4] 模型文件: {MODEL_PATH}")

# 检查是否已下载
if os.path.exists(MODEL_PATH):
    print(f"\n  模型文件已存在: {MODEL_PATH}")
    print("    如需重新下载，请删除 models/sam2/ 目录")
else:
    print(f"\n  [ ] 模型文件不存在，需要下载")

    # 选择下载方式
    print("\n[5] 选择下载方式:")
    print("    1. 自动下载（使用 urllib）")
    print("    2. 手动下载（请从浏览器下载）")

    choice = input("    请选择 (1/2，回车=手动): ").strip()

    if choice == "1":
        print("\n[6] 正在下载 SAM 2 模型...")
        download_auto()
    elif choice == "2":
        print("\n[ ] 手动下载说明:")
        print("    1. 访问以下链接:")
        print(f"       {SAM2_MODELS[MODEL_TYPE]}")
        print("    2. 下载后放到 models/sam2/ 目录")
        print("    3. 运行此脚本再次")
        return
    else:
        print("\n[ ] 已取消下载")


def download_auto():
    """自动下载 SAM 2 模型"""
    import urllib.request

    url = SAM2_MODELS[MODEL_TYPE]
    filename = f"sam2_hiera_{MODEL_TYPE}.pt"
    save_path = MODEL_PATH

    try:
        print(f"    正在下载: {url}")

        # 下载进度回调
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = int(downloaded / total_size * 100) if total_size > 0 else 0
            sys.stdout.write(f"\r        [{percent}%]")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, save_path, reporthook=report_progress)

        print(f"    [√] 下载完成: {save_path}")

    except Exception as e:
        print(f"    [×] 下载失败: {e}")
        print("    请检查网络连接或使用手动下载方式")

        return os.path.exists(save_path)


def create_config():
    """创建 SAM 2 配置文件"""
    config = f"""
# SAM 2 配置文件
# 自动生成 - {MODEL_TYPE} 版本

model_type: {MODEL_TYPE}
checkpoint_path: {MODEL_PATH}

# 编码器配置
encoder:
  type: sam2
  params:
    embed_dim: 384
  encoder_depth: 0   # tiny, small, base: 1, 2, 3
  mobile_sam: True  # 移动端支持

predictor:
  type: sam2
  params:
    min_mask_region_area: 0  # 返回最小区域掩码
    points_per_side: 32  # 输出点数
    points_per_batch: 64
    orig_im_size: 1024  # 输入图像大小
"""

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(config)
    print(f"    [√] 配置文件已创建: {CONFIG_PATH}")


def load_sam2_local():
    """测试本地 SAM 2 模型是否可用"""
    try:
        import torch
        from segment_anything_2 import sam_model_registry, SamModel
        from segment_anything_2.predictor import SamPredictor

        print("    [1] 检查 PyTorch...")
        if not torch.cuda.is_available():
            print("    [ ] CUDA 不可用，将使用 CPU")

        # 加载模型
        print(f"    [2] 加载 SAM 2 {MODEL_TYPE}...")

        # 注册模型
        sam = sam_model_registry[f"sam2_hiera_{MODEL_TYPE}"]()

        print(f"    [3] 模型已加载: {sam.__class__}")
        print(f"    [4] 模型类型: {type(sam)}")

        return True

    except ImportError as e:
        print(f"    [×] 缺少依赖: {e}")
        print("    请安装: pip install segment-anything")
        return False
    except Exception as e:
        print(f"    [×] 加载失败: {e}")
        return False


def test_sam2_segmentation():
    """测试 SAM 2 分割功能"""
    import torch
    import numpy as np
    from PIL import Image

    try:
        from segment_anything_2 import sam_model_registry, SamModel
        from segment_anything_2.predictor import SamPredictor

        sam = sam_model_registry["sam2_hiera_tiny"]()
        sam.eval()

        print("=" * 70)
        print("  SAM 2 本地模型测试")
        print("=" * 70)

        # 测试图片
        test_image = "data/R.jpg"
        if not os.path.exists(test_image):
            print(f"    [!] 测试图片不存在: {test_image}")
            print("    请先上传博物馆照片到 data/R.jpg")
            return False

        # 加载图片
        image = Image.open(test_image).convert("RGB")
        w, h = image.size
        print(f"    [1] 图片尺寸: {w} x {h}")

        # 准备输入
        from segment_anything_2.image_processor import SamImageProcessor

        processor = SamImageProcessor(sam.target_size)
        inputs = processor(image)

        print(f"    [2] 输入已准备")

        # 测试分割（单点）
        print("\n[3] 测试单点分割...")
        point_coords = [[w//2, h//2]]  # 中心点
        point_labels = [[1]]

        with torch.no_grad():
            outputs = sam(
                image_embeddings=inputs["image_embeddings"],
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        # 获取掩码
        masks = outputs.pred_masks.cpu().numpy()
        print(f"    [4] 输出掩码形状: {masks.shape}")

        # 获取边界框
        if len(outputs.pred_boxes) > 0:
            boxes = outputs.pred_boxes.cpu().numpy()
            print(f"    [5] 检测到 {len(boxes)} 个对象")

            # 获取IoU分数
        scores = outputs.iou_scores.cpu().numpy()
        print(f"    [6] IoU分数: {scores[:min(3, len(scores))]}")

        return True

    except Exception as e:
        print(f"    [×] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print(f"\nSAM 2 本地模型部署脚本")
    print(f"Python 版本: {sys.version.split()[0]}")

    # 检查依赖
    print("\n" + "=" * 70)
    print("[Step 0] 检查 segment-anything 包...")
    try:
        import segment_anything
        print("    [√] segment-anything 已安装")
    except ImportError:
        print("    [×] 缺少 segment-anything")
        print("    安装: pip install segment-anything")
        return

    # 下载模型
    print("\n" + "=" * 70)
    print("[Step 1] 下载 SAM 2 模型...")

    if download_auto():
        print("\n[Step 2] 创建配置文件...")
        create_config()
    else:
        print("\n[ ] 跳过下载步骤")

    # 测试模型加载
    print("\n" + "=" * 70)
    print("[Step 3] 测试本地 SAM 2 模型...")

    sam_loaded = load_sam2_local()

    if sam_loaded:
        print("\n" + "=" * 70)
        print("[Step 4] 测试 SAM 2 分割功能...")

        test_success = test_sam2_segmentation()

        if test_success:
            print("\n" + "=" * 70)
            print("[√] SAM 2 本地模型部署完成！")
            print("\n" + "=" * 70)
            print("  使用方法:")
            print("  from segment_anything_2 import sam_model_registry")
            print("  from segment_anything_2.predictor import SamPredictor")
            print("")
            print("  sam = sam_model_registry['sam2_hiera_tiny']()")
            print("  sam.eval()")
            print("")
            print("  # 加载图片:")
            print("  processor = SamImageProcessor(sam.target_size)")
            print("  inputs = processor(image)")
            print("")
            print("  # 分割:")
            print("  with torch.no_grad():")
            print("      outputs = sam(")
            print("        masks = outputs.pred_masks.cpu().numpy()")
            print("")
            print("  # 预测 (5个对象示例):")
            print("  point_coords = [[w//2, h//2]]")
            print("  point_labels = [[1]]")
            print("")
            print("  with torch.no_grad():")
            print("      outputs = sam(")
        else:
            print("  sam.eval()")
        else:
            print("  sam = sam_model_registry['sam2_hiera_tiny']()")
            print("")
            print("  # 加载图片:")
            print("  processor = SamImageProcessor(sam.target_size)")
            print("  inputs = processor(image)")
            print("")
            print("  # 分割:")
            print("  with torch.no_grad():")
            print("      outputs = sam(")
            print("        masks = outputs.pred_masks.cpu().numpy()")
            print("")
            print("  # 预测:")
            print("  point_coords = [[w//2, h//2]]")
            print("  point_labels = [[1]]")
            print("")
            print("  with torch.no_grad():")
            print("      outputs = sam(")
            print("        masks = outputs.pred_masks.cpu().numpy()")

            print("\n" + "=" * 70)
            print("  测试代码示例:")
            print("  from skills.visualization.pixel_heatmap import GazeHeatmapVisualizer")
            print("  viz = GazeHeatmapVisualizer('data/R.jpg')")
            print("  viz.add_vlm_regions([")
            print("      {'id': 'painting_1', 'label': '展品名称', 'type': 'Exhibit', 'bbox': SAM2返回的边界框}")
            print("  ])")
            print("  viz.add_gaze_record('painting_1', duration=45.0)")
            print("")
            print("  heatmap_data = viz.calculate_heatmap()")
            print("  viz.visualize_overlay(output_path='data/outputs/sam2_heatmap.png')")
        else:
            print("\n[ ] SAM 2 本地模型测试失败")

    print("\n" + "=" * 70)
    print("=" * 70)


if __name__ == "__main__":
    main()
