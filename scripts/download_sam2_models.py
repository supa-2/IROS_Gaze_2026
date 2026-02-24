#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 SAM2 模型权重

SAM2 模型选项:
- sam2_hiera_tiny.pt: ~38MB (最快，精度略低)
- sam2_hiera_small.pt: ~94MB (平衡)
- sam2_hiera_base_plus.pt: ~167MB
- sam2_hiera_large.pt: ~345MB (最精确，最慢)

对于本项目推荐使用 small 模型
"""

import os
import sys
import urllib.request
from pathlib import Path

# 项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 模型配置 - 使用 sam2.1 版本 (092824)
MODELS = {
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "size_mb": 39,
        "path": "models/sam2/sam2.1_hiera_tiny.pt",
        "config": "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
    },
    "small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "size_mb": 46,
        "path": "models/sam2/sam2.1_hiera_small.pt",
        "config": "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
    },
    "base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "size_mb": 81,
        "path": "models/sam2/sam2.1_hiera_base_plus.pt",
        "config": "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    },
    "large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "size_mb": 224,
        "path": "models/sam2/sam2.1_hiera_large.pt",
        "config": "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    }
}


def download_file(url, dest_path, description=""):
    """下载文件并显示进度"""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"[!] 文件已存在: {dest_path}")
        return True

    print(f"[*] 下载: {description}")
    print(f"    URL: {url}")
    print(f"    目标: {dest_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r    进度: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print(f"\n[+] 下载完成: {dest_path}")
        return True
    except Exception as e:
        print(f"\n[!] 下载失败: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="下载 SAM2 模型权重")
    parser.add_argument(
        "--model",
        type=str,
        choices=["tiny", "small", "base_plus", "large", "all"],
        default="small",
        help="要下载的模型 (默认: small)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: models/sam2/)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SAM2 模型下载器")
    print("=" * 60)

    if args.model == "all":
        models_to_download = list(MODELS.values())
    else:
        models_to_download = [MODELS[args.model]]

    # 调整输出目录
    if args.output_dir:
        for model in models_to_download:
            filename = Path(model["path"]).name
            model["path"] = str(Path(args.output_dir) / filename)

    # 下载模型
    success_count = 0
    for model in models_to_download:
        model_name = Path(model["path"]).stem
        if download_file(
            model["url"],
            os.path.join(project_root, model["path"]),
            f"{model_name} (~{model['size_mb']}MB)"
        ):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"完成: {success_count}/{len(models_to_download)} 个模型下载成功")
    print("=" * 60)

    if success_count > 0:
        print("\n[*] 使用方法:")
        print("    from skills.segmentation.sam2_local import SAM2LocalSegmenter")
        print("    segmenter = SAM2LocalSegmenter(model_path='models/sam2/sam2_hiera_small.pt')")


if __name__ == "__main__":
    main()
