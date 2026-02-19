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
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=500,    # Filter small regions
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

    # Sort by area and filter
    valid_masks = [m for m in masks if m.get('predicted_iou', 0) > 0.5]
    valid_masks.sort(key=lambda x: x['area'], reverse=True)

    # Keep top 15 by area
    valid_masks = valid_masks[:15]

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
        x1, y1, w, h = bbox
        box = (int(x1), int(y1), int(x1 + w), int(y1 + h))
        boxes.append(box)

        center_x = int(x1 + w / 2)
        center_y = int(y1 + h / 2)
        centers.append((center_x, center_y))

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
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        mask_img.save(mask_path)

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

    # Save centers for heatmap/trajectory
    centers_path = os.path.join(output_dir, "centers.json")
    with open(centers_path, 'w', encoding='utf-8') as f:
        json.dump({"centers": centers}, f, indent=2)
    print(f"  [*] Centers: {centers_path}")

    return output_dir


def create_overlay(image, masks, output_dir):
    """Create overlay visualization - only bounding boxes"""
    # Use distinctive colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 128), (128, 128, 0), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 128), (255, 255, 255)
    ]

    img_rgb = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    for i, mask_data in enumerate(masks):
        color = colors[i % len(colors)]
        bbox = mask_data.get('bbox', [0, 0, 0, 0])

        # Draw bbox with thick outline
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=4)

        # Draw label with background
        label = f"#{i+1}"
        text_bbox = draw.textbbox((x, y - 20), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x, y - 20), label, fill=(0, 0, 0))

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
