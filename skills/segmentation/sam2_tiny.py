# -*- coding: utf-8 -*-
"""
SAM 2 Tiny 分割 - 轻量级版本
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image

try:
    import torch
    from transformers import SamModel, SamProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SAM2Tiny:
    """SAM 2 Tiny 分割器"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.is_loaded = False
        
    def load(self) -> bool:
        if not TORCH_AVAILABLE:
            print("[Error] torch/transformers not installed")
            print("Install: pip install transformers torch")
            return False
        
        try:
            print("[SAM2] Loading sam2-hiera-tiny...")
            self.model = SamModel.from_pretrained("facebook/sam2-hiera-tiny")
            self.processor = SamProcessor.from_pretrained("facebook/sam2-hiera-tiny")
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print("[SAM2] Loaded successfully!")
            return True
        except Exception as e:
            print(f"[Error] {e}")
            return False
    
    def segment(self, image_path: str) -> List[Dict]:
        if not self.is_loaded:
            if not self.load():
                return [{"error": "Failed to load model"}]
        
        try:
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            masks = outputs.pred_masks.cpu().numpy()
            
            if len(masks.shape) == 4:
                masks = masks[0]
                masks = np.transpose(masks, (2, 0, 1))
            num_masks = masks.shape[0]
            
            results = []
            for i in range(num_masks):
                mask = masks[i]
                if not np.any(mask):
                    continue
                
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if not np.any(rows) or not np.any(cols):
                    continue
                
                y_idxs = np.where(rows)[0]
                x_idxs = np.where(cols)[0]
                y1, y2 = y_idxs[0], y_idxs[-1] + 1
                x1, x2 = x_idxs[0], x_idxs[-1] + 1
                
                area = (x2 - x1) * (y2 - y1)
                if area < 500:
                    continue
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                results.append({
                    "id": f"obj_{i}",
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "center": [int(cx), int(cy)],
                    "area": int(area)
                })
            
            return results
            
        except Exception as e:
            return [{"error": str(e)}]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sam2_tiny.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    sam = SAM2Tiny()
    results = sam.segment(image_path)
    
    print(f"Found {len(results)} objects:")
    for r in results:
        print(f"  {r}")
