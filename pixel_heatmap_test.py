import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def main():
    regions = [
        {"id": "ex_a", "label": "Bronze Ding", "bbox": [100, 150, 300, 350]},
        {"id": "ex_b", "label": "Pottery", "bbox": [400, 100, 550, 300]},
        {"id": "label_c", "label": "Sign", "bbox": [500, 350, 650, 450]}
    ]

    img = Image.open("data/test_museum.jpg")
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    colors = ['red', 'blue', 'green']
    for i, region in enumerate(regions):
        bbox = region["bbox"]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=colors[i], facecolor='none'
        )
        ax[0].add_patch(rect)
        ax[0].text(x1, y1 - 10, region["label"], fontsize=10, color='white')

    ax[0].set_title("VLM Identified Regions")

    plt.tight_layout()

    output_path = "data/outputs/pixel_heatmap_test.png"
    import os
    os.makedirs("data/outputs", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
