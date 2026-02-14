# -*- coding: utf-8 -*-
"""
SAM 2 本地模型下载脚本
"""

import os
import sys

# UTF-8 output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import urllib.request


def download_sam2_model():
    """下载 SAM 2 模型到本地"""
    # SAM 2 模型下载地址
    urls = {
        'tiny': 'https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2_hiera_tiny.pt',
        'small': 'https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2_hiera_small.pt',
        'base': 'https://github.com/facebookresearch/segment-anything-2/releases/download/v2.1.0/sam2_hiera_base.pt',
    }

    # 创建模型目录
    models_dir = 'models/sam2'
    os.makedirs(models_dir, exist_ok=True)

    # 选择版本（推荐 tiny）
    model_type = 'tiny'
    url = urls[model_type]
    save_path = os.path.join(models_dir, f'sam2_hiera_{model_type}.pt')

    print(f'[1] 下载 SAM 2 {model_type} 版本...')
    print(f'    URL: {url}')
    print(f'    保存: {save_path}')

    try:
        # 下载进度回调
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = int(downloaded / total_size * 100) if total_size > 0 else 0
            sys.stdout.write(f'\r    [{percent}%]')
            sys.stdout.flush()

        urllib.request.urlretrieve(url, save_path, reporthook=report_progress)

        file_size = os.path.getsize(save_path)
        print(f'[2] 下载完成! 文件大小: {file_size / 1024 / 1024:.1f} MB')
        return True

    except Exception as e:
        print(f'[×] 下载失败: {e}')
        return False


def main():
    print('=' * 70)
    print('  SAM 2 本地模型下载')
    print('=' * 70)

    success = download_sam2_model()

    print()
    print('=' * 70)
    if success:
        print('  ✓ 下载完成!')
        print()
        print('  下一步:')
        print('    1. 将 models/sam2/ 目录添加到 .gitignore')
        print('    2. 运行 heatmap/trajectory 生成脚本')
        print('       （将自动使用本地模型）')
    else:
        print('  × 下载失败')
        print()
        print('  建议:')
        print('    1. 检查网络连接')
        print('    2. 尝试从浏览器手动下载:')
        print(f'       {urls["tiny"]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
