#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 2 Small Model Calling Script
Automatic segmentation for all objects in image
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def load_sam2_small(model_path=None, config_path=None, device="auto"):
    """Load SAM 2 Small model with automatic mask generator"""
    if model_path is None:
        model_path = os.path.join(project_root, "models/sam2/sam2_hiera_small.pt")
    if config_path is None:
        config_path = "configs/sam2/sam2_hiera_s.yaml"

    print("=" * 60)
    print("SAM 2 Small Model Loading")
    print("=" * 60)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[Device] CUDA - {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("[Device] CPU")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[Model] {model_path}")
    print("\nLoading model...")

    sam2_model = build_sam2(
        config_file=config_path,
        ckpt_path=model_path,
        device=device,
    )

    # Create automatic mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,          # More points = more detections
        pred_iou_thresh=0.7,         # Only keep masks with IoU > 0.7
        stability_score_thresh=0.85, # Only keep stable masks
        min_mask_region_area=100,    # Filter small regions
        output_mode="binary_mask",
    )

    print("[+] Model loaded with AutomaticMaskGenerator!")
    return mask_generator


def segment_image(mask_generator, image_path):
    """Segment image automatically"""
    print(f"\n[Segment] Image: {image_path}")

    image = np.array(Image.open(image_path))
    print(f"[Size] {image.shape}")

    print("[*] Running automatic segmentation...")
    print("    (This may take 10-30 seconds depending on image size)")

    masks = mask_generator.generate(image)

    print(f"[+] Done! Found {len(masks)} objects")

    # Sort by area (largest first) and filter by confidence
    valid_masks = [m for m in masks if m.get('predicted_iou', 0) > 0.5]
    valid_masks.sort(key=lambda x: x['area'], reverse=True)

    print(f"    Valid masks (IoU > 0.5): {len(valid_masks)}")

    return {
        "masks": valid_masks,
        "image_shape": image.shape,
        "image_path": image_path,
        "original_image": image
    }


def extract_from_auto_masks(masks):
    """Extract boxes, centers, scores from auto masks"""
    boxes = []
    centers = []
    scores = []

    for mask_data in masks:
        bbox = mask_data.get('bbox', [0, 0, 0, 0])  # [x, y, w, h]
        # Convert to [x1, y1, x2, y2]
        x1, y1, w, h = bbox
        box = (int(x1), int(y1), int(x1 + w), int(y1 + h))
        boxes.append(box)

        # Center
        center_x = int(x1 + w / 2)
        center_y = int(y1 + h / 2)
        centers.append((center_x, center_y))

        # Score
        scores.append(mask_data.get('predicted_iou', 0.0))

    return boxes, centers, scores


def save_results(result, output_dir):
    """Save segmentation results"""
    os.makedirs(output_dir, exist_ok=True)

    masks = result.get('masks', [])
    image = result.get('original_image')

    if not masks:
        print("[!] No masks found!")
        return output_dir

    # Save individual masks
    for i, mask_data in enumerate(masks[:20]):  # Max 20 masks
        mask = mask_data['segmentation']
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        mask_img.save(mask_path)

        # Get bbox and score for display
        bbox = mask_data.get('bbox', [0, 0, 0, 0])
        score = mask_data.get('predicted_iou', 0.0)
        area = mask_data.get('area', 0)
        print(f"  [*] Mask {i}: bbox={bbox}, area={area}, iou={score:.3f}")

    # Extract info
    boxes, centers, scores = extract_from_auto_masks(masks)

    # Save JSON
    detections = []
    for i, mask_data in enumerate(masks):
        detections.append({
            "id": i,
            "bbox": boxes[i] if i < len(boxes) else None,
            "center": centers[i] if i < len(centers) else None,
            "confidence": float(scores[i]) if i < len(scores) else None,
            "area": int(mask_data.get('area', 0))
        })

    info = {
        "image_path": result.get("image_path"),
        "image_shape": result.get("image_shape"),
        "num_masks": len(masks),
        "detections": detections
    }

    info_path = os.path.join(output_dir, "segmentation_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  [*] Info: {info_path}")

    # Create overlay
    create_overlay(image, masks, output_dir)

    return output_dir


def create_overlay(image, masks, output_dir):
    """Create overlay visualization"""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
    ]

    img_rgb = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    for i, mask_data in enumerate(masks[:len(colors)]):
        color = colors[i % len(colors)]
        mask = mask_data['segmentation']
        bbox = mask_data.get('bbox', [0, 0, 0, 0])

        # Draw bbox
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        # Draw mask (semi-transparent)
        mask_img = Image.fromarray((mask * 128).astype(np.uint8), mode='L')
        mask_rgba = Image.new("RGBA", mask_img.size, color + (128,))
        mask_rgba = Image.alpha_composite(mask_rgba.convert("RGBA"), mask_img.convert("RGBA"))

        # Blend with original
        img_rgb_rgba = img_rgb.convert("RGBA")
        img_rgb_rgba = Image.alpha_composite(img_rgb_rgba, mask_rgba)
        img_rgb = img_rgb_rgba.convert("RGB")

        # Draw label
        label = f"{i} (IoU:{mask_data.get('predicted_iou', 0):.2f})"
        draw.text((x, y - 15), label, fill=color)

    # Save overlay
    overlay_path = os.path.join(output_dir, "overlay.png")
    img_rgb.save(overlay_path)
    print(f"  [*] Overlay: {overlay_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM 2 Automatic Segmentation")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--output", type=str, default="data/outputs/sam2_small", help="Output directory")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--config", type=str, default=None, help="Config path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"[!] Error: Image not found: {args.image}")
            return
        image_path = args.image
    else:
        image_path = "data/R.jpg"
        if not os.path.exists(image_path):
            print(f"[!] Default image not found: {image_path}")
            return

    try:
        mask_generator = load_sam2_small(args.model, args.config, args.device)
        result = segment_image(mask_generator, image_path)
        save_results(result, args.output)
        print(f"\n[Done] Results saved to: {args.output}/")

    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
