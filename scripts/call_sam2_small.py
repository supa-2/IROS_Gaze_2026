#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 2 Small Model Calling Script
For servers with sam2-hiera-small deployed
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add sam2 module path (already installed via setup_sam2.py)
import torch
import numpy as np
from PIL import Image, ImageDraw

# Import from installed sam2 package
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_sam2_small(model_path=None, config_path=None, device="auto"):
    """
    Load SAM 2 Small model

    Args:
        model_path: Model file path (auto-search if None)
        config_path: Config file path for Hydra (use relative path like "configs/sam2/sam2_hiera_s.yaml")
        device: Device selection ("auto", "cuda", "cpu")

    Returns:
        SAM2ImagePredictor instance
    """
    # Get project root
    if model_path is None:
        model_path = os.path.join(project_root, "models/sam2/sam2_hiera_small.pt")
    if config_path is None:
        # Use relative path for Hydra (relative to sam2 package)
        config_path = "configs/sam2/sam2_hiera_s.yaml"

    print("=" * 60)
    print("SAM 2 Small Model Loading")
    print("=" * 60)

    # Auto detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[Device] CUDA - {torch.cuda.get_device_name(0)}")
            print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            print("[Device] CPU")

    # Check model file existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[Model] {model_path}")
    print(f"[Config] {config_path} (Hydra relative path)")

    # Load model using Hydra config path
    print("\nLoading model...")
    sam2_model = build_sam2(
        config_file=config_path,
        ckpt_path=model_path,
        device=device,
    )

    print("[+] Model loaded successfully!")

    # Create predictor
    predictor = SAM2ImagePredictor(sam2_model)
    print("[+] Predictor created successfully!")

    return predictor


def segment_image(predictor, image_path, auto_segment=True):
    """Segment image"""
    print(f"\n[Segment] Image: {image_path}")

    image = np.array(Image.open(image_path))
    print(f"[Size] {image.shape}")

    predictor.set_image(image)
    print("[+] Image set")

    if auto_segment:
        print("[Mode] Auto segment")
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=True,
        )
    else:
        print("[Mode] Manual point segment")
        h, w = image.shape[:2]
        point_coords = np.array([[w//2, h//2]])
        point_labels = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    print(f"[+] Segment done! Detected {len(masks)} masks")
    print(f"   Confidence range: {scores.min():.3f} ~ {scores.max():.3f}")

    boxes, centers = extract_boxes_and_centers(masks)

    return {
        "masks": masks,
        "scores": scores.tolist(),
        "logits": logits.tolist(),
        "boxes": boxes,
        "centers": centers,
        "image_shape": image.shape,
        "image_path": image_path
    }


def extract_boxes_and_centers(masks):
    """Extract boxes and centers from masks"""
    boxes = []
    centers = []

    for i, mask in enumerate(masks):
        if isinstance(mask, list):
            mask_array = np.array(mask)
        else:
            mask_array = mask

        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)

        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            boxes.append((int(cmin), int(rmin), int(cmax), int(rmax)))

            center_x = int(np.mean([cmin, cmax]))
            center_y = int(np.mean([rmin, rmax]))
            centers.append((center_x, center_y))
        else:
            boxes.append((0, 0, 0, 0))
            centers.append((0, 0))

    return boxes, centers


def save_results(result, output_dir, show_overlay=True):
    """Save segmentation results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save mask images
    for i, mask in enumerate(result.get('masks', [])):
        if isinstance(mask, list):
            mask_array = np.array(mask, dtype=np.uint8) * 255
        else:
            mask_array = (mask * 255).astype(np.uint8)

        mask_img = Image.fromarray(mask_array, mode='L')
        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        mask_img.save(mask_path)
        print(f"  [*] Mask {i}: {mask_path}")

    # Save segmentation info
    boxes = result.get('boxes', [])
    centers = result.get('centers', [])
    scores = result.get('scores', [])

    detections = []
    for i in range(len(boxes)):
        detections.append({
            "id": i,
            "bbox": boxes[i] if i < len(boxes) else None,
            "center": centers[i] if i < len(centers) else None,
            "confidence": float(scores[i]) if i < len(scores) else None
        })

    info = {
        "image_path": result.get("image_path"),
        "image_shape": result.get("image_shape"),
        "num_masks": len(result.get('masks', [])),
        "detections": detections
    }

    info_path = os.path.join(output_dir, "segmentation_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  [*] Info: {info_path}")

    if show_overlay and result.get('boxes'):
        create_overlay(result, output_dir)

    return output_dir


def create_overlay(result, output_dir):
    """Create segmentation result overlay image"""
    image_path = result.get("image_path")
    if not os.path.exists(image_path):
        print("  [!] Original image not found, skip overlay")
        return

    image = Image.open(image_path).convert("RGBA")
    masks = result.get('masks', [])

    colors = [
        (255, 0, 0, 128),
        (0, 255, 0, 128),
        (0, 0, 255, 128),
        (255, 255, 0, 128),
        (255, 0, 255, 128),
    ]

    for i, mask in enumerate(masks[:len(colors)]):
        color = colors[i % len(colors)]
        if isinstance(mask, list):
            mask_array = np.array(mask)
        else:
            mask_array = mask

        mask_rgba = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        mask_rgba[mask_array] = color

        mask_img = Image.fromarray(mask_rgba, 'RGBA')
        image = Image.alpha_composite(image.convert('RGBA'), mask_img)

    overlay_path = os.path.join(output_dir, "overlay.png")
    image.convert('RGB').save(overlay_path)
    print(f"  [*] Overlay: {overlay_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM 2 Small Model Calling")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--output", type=str, default="data/outputs/sam2_small", help="Output directory")
    parser.add_argument("--model", type=str, default=None, help="Model path (auto-search if not specified)")
    parser.add_argument("--config", type=str, default=None, help="Config path (default: configs/sam2/sam2_hiera_s.yaml)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    parser.add_argument("--manual", action="store_true", help="Use manual point segmentation mode")

    args = parser.parse_args()

    # Check image
    if args.image:
        if not os.path.exists(args.image):
            print(f"[!] Error: Image not found: {args.image}")
            return
        image_path = args.image
    else:
        image_path = "data/R.jpg"
        if not os.path.exists(image_path):
            print(f"[!] Default image not found: {image_path}")
            print("Please use --image to specify image path")
            return

    # Load model
    try:
        predictor = load_sam2_small(args.model, args.config, args.device)
    except Exception as e:
        print(f"[!] Model load failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure sam2 is installed: python setup_sam2.py")
        print("2. Check config file exists: sam2/configs/sam2/sam2_hiera_s.yaml")
        print("3. Check model file exists: models/sam2/sam2_hiera_small.pt")
        return

    # Execute segmentation
    result = segment_image(predictor, image_path, auto_segment=not args.manual)

    # Save results
    print(f"\n[Save Results]")
    save_results(result, args.output)

    print(f"\n[Done] Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
