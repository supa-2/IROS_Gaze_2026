# -*- coding: utf-8 -*-
"""SAM 2 Tiny 测试"""
import os, sys, time
sys.path.insert(0, os.getcwd())

try:
    import torch
    from transformers import SamModel, SamProcessor

    print("Loading SAM 2 Tiny...")
    model = SamModel.from_pretrained("facebook/sam2-hiera-tiny")
    processor = SamProcessor.from_pretrained("facebook/sam2-hiera-tiny")
    print(f"Model loaded! Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # 测试分割
    from PIL import Image
    import numpy as np

    image_path = "data/R.jpg"
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    inputs = processor(img, return_tensors="pt")
    outputs = model(**inputs)
    masks = outputs.pred_masks[0].cpu().numpy()

    print(f"Image size: {w} x {h}")
    print(f"Masks shape: {masks.shape}")
    print("✅ SAM 2 Tiny 测试成功!")

except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("安装: uv pip install transformers torch")
except Exception as e:
    print(f"❌ 错误: {e}")
