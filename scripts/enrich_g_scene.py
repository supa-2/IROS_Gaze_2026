#!/usr/bin/env python3
"""
为G场景生成详细的图片分析JSON数据
"""
import json
from pathlib import Path

# G场景的VLM详细分析数据
g_scene_analysis = {
    "scene_info": {
        "scene": "G",
        "type": "Chinese Traditional Art Gallery",
        "layout": "Circular gallery with central cylindrical pedestal",
        "environment": {
            "wall_color": "White",
            "floor": "Reflective smooth surface",
            "lighting": "Black ring-shaped ceiling light with white inner ring",
            "pedestal": "White cylindrical display pedestal with circular glowing base"
        }
    },
    "exhibits": [
        {
            "id": "E1",
            "name": "Flower and Bird Painting",
            "type": "Painting",
            "style": "Chinese Ink Painting (Xieyi)",
            "description": "Bird and flower theme, expressing natural vitality through wash techniques",
            "colors": ["Warm Yellow", "Light Green", "Off-White"],
            "position": "Left Wall - Position 1 (smallest size)",
            "bbox_color": "Blue",
            "gaze_data": {
                "duration": 6.25,
                "attention": "Low",
                "heatmap_pattern": "Small red circle (high intensity, small range)"
            }
        },
        {
            "id": "E2",
            "name": "Landscape Painting",
            "type": "Painting",
            "style": "Chinese Ink Landscape",
            "description": "Mountains, water and vegetation with layered ink showing spatial depth",
            "colors": ["Dark Green", "Light Blue", "Grey-White"],
            "position": "Left Wall - Position 2",
            "bbox_color": "Green",
            "gaze_data": {
                "duration": 6.57,
                "attention": "Medium",
                "heatmap_pattern": "Medium red circle (high intensity, medium range)"
            }
        },
        {
            "id": "E3",
            "name": "Architecture and Figure",
            "type": "Painting",
            "style": "Chinese Ink Painting (Gongbi)",
            "description": "Traditional Chinese architecture with figure, possibly historical scene",
            "colors": ["Black", "White", "Light Red"],
            "position": "Center - Pedestal (focal point)",
            "bbox_color": "Brown",
            "gaze_data": {
                "duration": 6.75,
                "attention": "High",
                "heatmap_pattern": "Red-yellow gradient circle (medium intensity, medium range)"
            }
        },
        {
            "id": "E4",
            "name": "Ink Landscape",
            "type": "Painting",
            "style": "Chinese Ink Landscape (Xieyi)",
            "description": "Black, grey and white tones depicting mountains and mist",
            "colors": ["Black", "Grey", "White"],
            "position": "Right Wall - Position 1",
            "bbox_color": "Grey",
            "gaze_data": {
                "duration": 6.81,
                "attention": "High",
                "heatmap_pattern": "Blue-red gradient circle (lower intensity, larger range)"
            }
        },
        {
            "id": "E5",
            "name": "Bird and Flower Mood",
            "type": "Painting",
            "style": "Chinese Ink Painting (Xieyi)",
            "description": "Trees and flying birds in soft blue and yellow tones",
            "colors": ["Light Blue", "Bright Yellow", "Off-White"],
            "position": "Right Wall - Position 2",
            "bbox_color": "Cyan",
            "gaze_data": {
                "duration": 6.41,
                "attention": "Medium-Low",
                "heatmap_pattern": "Red-yellow gradient circle (medium intensity, medium range)"
            }
        }
    ],
    "panel_analysis": {
        "panel_a": {
            "title": "Original Scene",
            "description": "Modern art gallery with 5 traditional Chinese paintings arranged in circular layout",
            "resolution": "1100 × 618 px",
            "key_features": [
                "Central white cylindrical pedestal",
                "5 paintings evenly distributed on curved wall",
                "Black ring-shaped ceiling light",
                "Reflective flooring"
            ]
        },
        "panel_b": {
            "title": "Segmentation Result",
            "description": "SAM2-based exhibit detection with color-coded bounding boxes",
            "detection_count": 5,
            "bbox_format": "E[N] P (Exhibit Number Painting)",
            "color_coding": {
                "E1": "Blue",
                "E2": "Green",
                "E3": "Brown",
                "E4": "Grey",
                "E5": "Cyan"
            }
        },
        "panel_c": {
            "title": "Attention Heatmap",
            "description": "Gaussian kernel density estimation with center-hotspot pattern",
            "color_scale": "Blue (low) → Green → Yellow → Red (high)",
            "hotspot_pattern": "Central hotspot with gradient falloff",
            "highest_attention": ["E3", "E4"],
            "lowest_attention": ["E1"]
        },
        "panel_d": {
            "title": "Gaze Trajectory",
            "description": "Temporal scan path with sequential fixation points",
            "fixation_count": 5,
            "scan_pattern": "Linear left-to-right (1→2→3→4→5)",
            "path_type": "Polyline connecting numbered fixation points",
            "fixation_markers": "Yellow circles with sequence numbers"
        }
    },
    "statistics": {
        "total_exhibits": 5,
        "gazed_exhibits": 5,
        "total_fixations": 5,
        "total_duration": 32.78,
        "mean_duration": 6.56,
        "std_duration": 0.21,
        "min_duration": 6.25,
        "max_duration": 6.81,
        "scan_pattern": "Linear left-to-right"
    }
}

# 读取原始数据并合并
def create_enriched_g_json():
    base_dir = Path("/home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026")

    # 读取原始数据
    with open(base_dir / "data/outputs/G/table_data.json", 'r') as f:
        original_data = json.load(f)

    # 创建增强数据
    enriched_data = {
        "scene": "G",
        "timestamp": original_data.get("timestamp", ""),
        "scene_info": g_scene_analysis["scene_info"],
        "exhibits": [],
        "panel_analysis": g_scene_analysis["panel_analysis"],
        "statistics": g_scene_analysis["statistics"],
        "table_data": {
            "columns": ["ID", "Name", "Type", "Style", "Location", "Duration (s)", "Attention"],
            "rows": []
        }
    }

    attention_map = {"A": "High", "B": "Medium-High", "C": "Medium", "D": "Medium-Low", "E": "Low"}

    # 合并展品数据
    for orig_exhibit in original_data["exhibits"]:
        eid = orig_exhibit["id"]
        vlm_exhibit = next((e for e in g_scene_analysis["exhibits"] if e["id"] == eid), {})

        exhibit_enriched = {
            "id": eid,
            "name": vlm_exhibit.get("name", orig_exhibit["name"]),
            "type": orig_exhibit["type"],
            "description": vlm_exhibit.get("description", ""),
            "style": vlm_exhibit.get("style", ""),
            "colors": vlm_exhibit.get("colors", []),
            "location": vlm_exhibit.get("position", ""),
            "bbox": orig_exhibit["bbox"],
            "center": orig_exhibit["center"],
            "area": orig_exhibit["area"],
            "sam_score": orig_exhibit["sam_score"],
            "gaze_count": orig_exhibit["gaze_count"],
            "total_duration": orig_exhibit["total_duration"],
            "avg_duration": orig_exhibit["avg_duration"],
            "attention_level": orig_exhibit["attention_level"],
            "attention_label": attention_map.get(orig_exhibit["attention_level"], ""),
            "bbox_color": vlm_exhibit.get("bbox_color", ""),
            "heatmap_pattern": vlm_exhibit.get("gaze_data", {}).get("heatmap_pattern", "")
        }
        enriched_data["exhibits"].append(exhibit_enriched)

        # 添加表格行
        enriched_data["table_data"]["rows"].append({
            "ID": eid,
            "Name": exhibit_enriched["name"],
            "Type": exhibit_enriched["type"],
            "Style": exhibit_enriched["style"],
            "Location": exhibit_enriched["location"],
            "Duration (s)": round(orig_exhibit["total_duration"], 2),
            "Attention": exhibit_enriched["attention_label"]
        })

    # 合并注视序列数据
    enriched_data["fixations"] = []
    for fixation in original_data["fixations"]:
        eid = fixation["exhibit_id"]
        vlm_exhibit = next((e for e in g_scene_analysis["exhibits"] if e["id"] == eid), {})

        fixation_enriched = {
            "sequence": fixation["sequence"],
            "exhibit_id": fixation["exhibit_id"],
            "exhibit_name": fixation["exhibit_name"],
            "center": fixation["center"],
            "duration": fixation["duration"],
            "score": fixation["score"],
            "description": vlm_exhibit.get("description", ""),
            "style": vlm_exhibit.get("style", "")
        }
        enriched_data["fixations"].append(fixation_enriched)

    # 保存文件
    output_path = base_dir / "data/outputs/G/scene_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    return enriched_data

if __name__ == "__main__":
    data = create_enriched_g_json()

    print("="*100)
    print(" G SCENE - ENRICHED ANALYSIS DATA")
    print("="*100)
    print(f"\nScene Type: {data['scene_info']['type']}")
    print(f"Layout: {data['scene_info']['layout']}")
    print(f"\nExhibits: {data['statistics']['total_exhibits']}")
    print(f"Total Duration: {data['statistics']['total_duration']}s")
    print(f"Mean Duration: {data['statistics']['mean_duration']}s ± {data['statistics']['std_duration']}s")

    print("\n" + "-"*100)
    print(f"{'ID':<6} {'Name':<25} {'Style':<25} {'Duration':<10} {'Attention':<12}")
    print("-"*100)
    for row in data["table_data"]["rows"]:
        print(f"{row['ID']:<6} {row['Name']:<25} {row['Style']:<25} {row['Duration (s)']:<10} {row['Attention']:<12}")

    print("\n" + "="*100)
    print("FIGURE CAPTION FOR SCENE G:")
    print("="*100)
    print("""
Figure X: Visual attention analysis of Chinese traditional art gallery. (a) Original exhibition
scene (1100×618px) featuring 5 ink paintings arranged on a curved wall surrounding a central
cylindrical white pedestal. (b) SAM2 segmentation results with color-coded bounding boxes
(E1=Blue, E2=Green, E3=Brown, E4=Grey, E5=Cyan) showing detected exhibit regions. (c) Attention
heatmap with Gaussian kernel density estimation reveals center-focused gaze distribution across
all paintings. (d) Gaze trajectory (1→2→3→4→5) demonstrating linear left-to-right scanning
pattern. Total viewing duration: 32.78s (mean: 6.56s ± 0.21s), with highest attention on E3
(6.75s) and E4 (6.81s).
    """.strip())

    print("\nFile saved to: data/outputs/G/scene_analysis.json")
