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

# Add sam2 module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam2'))

import torch
import numpy as np
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_sam2_small(model_path="models/sam2/sam2_hiera_small.pt",
                    config_path="sam2/configs/sam2/sam2_hiera_s.yaml",
                    device="auto"):
    """
    Load SAM 2 Small model

    Args:
        model_path: Model file path
        config_path: Config file path
        device: Device selection ("auto", "cuda", "cpu")

    Returns:
        SAM2ImagePredictor instance
    """
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

    # Check file existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"[Model] {model_path}")
    print(f"[Config] {config_path}")

    # Load model
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
    """
    Segment image

    Args:
        predictor: SAM2 predictor
        image_path: Image path
        auto_segment: Whether to use auto segmentation

    Returns:
        Segmentation result dict
    """
    print(f"\n[Segment] Image: {image_path}")

    # Load image
    image = np.array(Image.open(image_path))
    print(f"[Size] {image.shape}")

    # Set image
    predictor.set_image(image)
    print("[+] Image set")

    if auto_segment:
        # Auto segment
        print("[Mode] Auto segment")
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=True,
        )
    else:
        # Manual point segment
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

    # Extract boxes and centers
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

        # Calculate bbox
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)

        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            boxes.append((int(cmin), int(rmin), int(cmax), int(rmax)))

            # Calculate center
            center_x = int(np.mean([cmin, cmax]))
            center_y = int(np.mean([rmin, rmax]))
            centers.append((center_x, center_y))
        else:
            boxes.append((0, 0, 0, 0))
            centers.append((0, 0))

    return boxes, centers


def save_results(result, output_dir, show_overlay=True):
    """Save segmentation results to disk."""
    import os
    from PIL import Image

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

    # Create overlay visualization
    if show_overlay and result.get('boxes'):
        create_overlay(result, output_dir)

    return output_dir


def create_overlay(result, output_dir):
    """Create segmentation result overlay image"""
    from PIL import Image

    image_path = result.get("image_path")
    if not os.path.exists(image_path):
        print("  [!] Original image not found, skip overlay")
        return

    image = Image.open(image_path).convert("RGBA")
    masks = result.get('masks', [])

    # Assign colors for each mask
    colors = [
        (255, 0, 0, 128),    # Red
        (0, 255, 0, 128),    # Green
        (0, 0, 255, 128),    # Blue
        (255, 255, 0, 128),  # Yellow
        (255, 0, 255, 128),  # Magenta
    ]

    for i, mask in enumerate(masks[:len(colors)]):
        color = colors[i % len(colors)]
        if isinstance(mask, list):
            mask_array = np.array(mask)
        else:
            mask_array = mask

        # Create RGBA mask
        mask_rgba = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        mask_rgba[mask_array] = color

        mask_img = Image.fromarray(mask_rgba, 'RGBA')

        # Composite onto original image
        image = Image.alpha_composite(image.convert('RGBA'), mask_img)

    # Save overlay
    overlay_path = os.path.join(output_dir, "overlay.png")
    image.convert('RGB').save(overlay_path)
    print(f"  [*] Overlay: {overlay_path}")


def find_model_and_config():
    """Auto-find model and config files in common paths."""
    # Possible model paths (try both underscore and hyphen naming)
    model_candidates = [
        "models/sam2/sam2_hiera_small.pt",
        "models/sam2/sam2-hiera-small.pt",
        "models/sam2_hiera_small.pt",
        "models/sam2-hiera-small.pt",
        "../models/sam2_hiera_small.pt",
        "../models/sam2-hiera-small.pt",
        "../../models/sam2_hiera_small.pt",
        "~/models/sam2_hiera_small.pt",
        "~/models/sam2-hiera-small.pt",
        "/opt/models/sam2_hiera_small.pt",
    ]

    # Possible config paths
    config_candidates = [
        "sam2/configs/sam2/sam2_hiera_s.yaml",
        "sam2/configs/sam2-hiera-small.yaml",
        "sam2/configs/sam2_hiera_s.yaml",
        "sam2_hiera_s.yaml",
        "configs/sam2/sam2_hiera_s.yaml",
    ]

    model_path = None
    config_path = None

    for candidate in model_candidates:
        expanded = os.path.expanduser(candidate)
        if os.path.exists(expanded):
            model_path = expanded
            break

    for candidate in config_candidates:
        expanded = os.path.expanduser(candidate)
        if os.path.exists(expanded):
            config_path = expanded
            break

    return model_path, config_path


def main():
    parser = argparse.ArgumentParser(description="SAM 2 Small Model Calling")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--output", type=str, default="data/outputs/sam2_small", help="Output directory")
    parser.add_argument("--model", type=str, default=None, help="Model path (auto-search if not specified)")
    parser.add_argument("--config", type=str, default=None, help="Config file path (auto-search if not specified)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    parser.add_argument("--manual", action="store_true", help="Use manual point segmentation mode")

    args = parser.parse_args()

    # Auto-find model and config if not specified
    if args.model is None or args.config is None:
        found_model, found_config = find_model_and_config()
        if args.model is None:
            args.model = found_model if found_model else "models/sam2/sam2_hiera_small.pt"
        if args.config is None:
            args.config = found_config if found_config else "sam2/configs/sam2/sam2_hiera_s.yaml"

    # Check image
    if args.image:
        if not os.path.exists(args.image):
            print(f"[!] Error: Image not found: {args.image}")
            return
        image_path = args.image
    else:
        # Use default test image
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
        print("\nPlease confirm these files exist:")
        print(f"  - Model: {args.model}")
        print(f"  - Config: {args.config}")
        print("\nIf config is missing, run:")
        print("  cd ~ && git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo")
        print("  cp -r sam2_repo/sam2/configs ./sam2/")
        return

    # Execute segmentation
    result = segment_image(predictor, image_path, auto_segment=not args.manual)

    # Save results
    print(f"\n[Save Results]")
    save_results(result, args.output)

    print(f"\n[Done] Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
