#!/usr/bin/env python3
"""
为R场景生成详细的图片分析JSON数据
"""
import json
from pathlib import Path

# R场景的VLM详细分析数据
r_scene_analysis = {
    "scene_info": {
        "scene": "R",
        "type": "Modern Contemporary Art Gallery",
        "layout": "Linear gallery with left wall, center pedestal, and right wall",
        "environment": {
            "wall_color": "White",
            "floor": "Grey carpet",
            "lighting": "White track lighting from ceiling",
            "furniture": "Black chair in left-center area"
        }
    },
    "exhibits": [
        {
            "id": "E1",
            "name": "Landscape Abstraction",
            "type": "Painting",
            "style": "Abstract/Minimalist",
            "description": "Green gradient with black tones, abstract natural landscape (grassland or water)",
            "colors": ["Green", "Black", "Dark Green"],
            "position": "Left Wall - Position 1",
            "frame": "Black frame",
            "content_type": "Abstract Landscape",
            "gaze_data": {
                "duration": 4.97,
                "attention": "Low",
                "sequence": 1
            }
        },
        {
            "id": "E2",
            "name": "Portrait Study",
            "type": "Painting",
            "style": "Contemporary/Minimalist",
            "description": "Grey background with a minimalist black human portrait (short hair, simple facial details)",
            "colors": ["Grey", "Black", "White"],
            "position": "Left Wall - Position 2",
            "frame": "Black frame",
            "content_type": "Portrait",
            "gaze_data": {
                "duration": 5.54,
                "attention": "Medium-Low",
                "sequence": 2
            }
        },
        {
            "id": "E3",
            "name": "Dark Composition",
            "type": "Painting",
            "style": "Abstract/Experimental",
            "description": "Dark abstract composition with black and dark grey tones, possibly lines or geometric shapes",
            "colors": ["Black", "Dark Grey", "Charcoal"],
            "position": "Left Wall - Position 3",
            "frame": "Black frame",
            "content_type": "Abstract",
            "gaze_data": {
                "duration": 5.84,
                "attention": "Medium",
                "sequence": 4
            }
        },
        {
            "id": "E4",
            "name": "Light Minimalist",
            "type": "Painting",
            "style": "Minimalist/Contemporary",
            "description": "Light-toned minimalist work with light grey and off-white colors",
            "colors": ["Light Grey", "Off-White", "Beige"],
            "position": "Left Wall - Position 4",
            "frame": "Black frame",
            "content_type": "Abstract Minimalist",
            "gaze_data": {
                "duration": 6.25,
                "attention": "Medium-High",
                "sequence": 3
            }
        },
        {
            "id": "E5",
            "name": "Central Sculpture",
            "type": "Sculpture",
            "style": "Contemporary Sculpture",
            "description": "Abstract form on white cubic pedestal, organic/geometric hybrid shape with curves",
            "colors": ["Grey", "Stone"],
            "material": "Ceramic or Stone",
            "position": "Center - Pedestal",
            "content_type": "Abstract Sculpture",
            "gaze_data": {
                "duration": 6.07,
                "attention": "Medium-High",
                "sequence": 5
            }
        },
        {
            "id": "E6",
            "name": "Sculpture Base Area",
            "type": "Sculpture",
            "style": "Contemporary",
            "description": "Sculpture pedestal and surrounding area",
            "colors": ["White", "Grey"],
            "material": "White painted wood/pedestal",
            "position": "Center - Pedestal",
            "content_type": "Display Element",
            "gaze_data": {
                "duration": 5.72,
                "attention": "Medium",
                "sequence": 6
            }
        },
        {
            "id": "E7",
            "name": "Silhouette Landscape",
            "type": "Painting",
            "style": "Photographic/Minimalist",
            "description": "Blue sky gradient (light to dark blue) with black silhouette of person standing in grass with vegetation",
            "colors": ["Blue", "Black", "Light Blue"],
            "position": "Right Wall - Main",
            "frame": "Black frame",
            "content_type": "Landscape with Figure",
            "gaze_data": {
                "duration": 5.67,
                "attention": "Medium",
                "sequence": 9
            }
        },
        {
            "id": "E8",
            "name": "Upper Wall Art A",
            "type": "Painting",
            "style": "Contemporary",
            "description": "Upper wall display element",
            "colors": ["Mixed"],
            "position": "Upper Right Wall",
            "frame": "Black frame",
            "content_type": "Wall Art",
            "gaze_data": {
                "duration": 6.69,
                "attention": "High",
                "sequence": 7
            }
        },
        {
            "id": "E9",
            "name": "Upper Wall Art B",
            "type": "Painting",
            "style": "Contemporary",
            "description": "Secondary upper wall display",
            "colors": ["Mixed"],
            "position": "Upper Right Wall",
            "frame": "Black frame",
            "content_type": "Wall Art",
            "gaze_data": {
                "duration": 6.60,
                "attention": "High",
                "sequence": 8
            }
        }
    ],
    "panel_analysis": {
        "panel_a": {
            "title": "Original Scene",
            "description": "Modern contemporary art gallery with white walls, grey floor, track lighting",
            "resolution": "1100 × 600 px",
            "key_features": [
                "Left wall: 4 paintings in black frames",
                "Center: White cubic pedestal with abstract sculpture",
                "Right wall: Large blue-toned painting with figure silhouette",
                "Upper right wall: 2 additional display elements",
                "Black chair in left-center area"
            ]
        },
        "panel_b": {
            "title": "Segmentation Result",
            "description": "SAM2-based exhibit detection with colored region masks and bounding boxes",
            "detection_count": 9,
            "method": "SAM2 (Segment Anything Model 2)",
            "output_format": "Color-coded masks with bounding boxes and center points"
        },
        "panel_c": {
            "title": "Attention Heatmap",
            "description": "Gaussian kernel density estimation of gaze fixation distribution",
            "color_scale": "Blue (low) → Green → Yellow → Red (high)",
            "hotspot_pattern": "Multiple hotspots corresponding to exhibit locations",
            "highest_attention": ["E8 (6.69s)", "E9 (6.60s)", "E4 (6.25s)"],
            "lowest_attention": ["E1 (4.97s)"]
        },
        "panel_d": {
            "title": "Gaze Trajectory",
            "description": "Temporal scan path showing sequential fixation order",
            "fixation_count": 9,
            "scan_pattern": "Left-to-right with center branching (1→2→4→3→5→6→7→8→9)",
            "path_type": "Polyline connecting numbered fixation points",
            "fixation_markers": "Yellow circles with sequence numbers 1-9"
        }
    },
    "statistics": {
        "total_exhibits": 9,
        "gazed_exhibits": 9,
        "total_fixations": 9,
        "total_duration": 53.34,
        "mean_duration": 5.93,
        "std_duration": 0.51,
        "min_duration": 4.97,
        "max_duration": 6.69,
        "scan_pattern": "Left-to-Right with Center Anchor"
    }
}

def create_enriched_r_json():
    base_dir = Path("/home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026")

    # 读取原始数据
    with open(base_dir / "data/outputs/R/table_data.json", 'r') as f:
        original_data = json.load(f)

    # 创建增强数据
    enriched_data = {
        "scene": "R",
        "timestamp": original_data.get("timestamp", ""),
        "scene_info": r_scene_analysis["scene_info"],
        "exhibits": [],
        "panel_analysis": r_scene_analysis["panel_analysis"],
        "statistics": r_scene_analysis["statistics"],
        "table_data": {
            "columns": ["ID", "Name", "Type", "Style", "Location", "Duration (s)", "Attention"],
            "rows": []
        }
    }

    attention_map = {"A": "High", "B": "Medium-High", "C": "Medium", "D": "Medium-Low", "E": "Low"}

    # 合并展品数据
    for orig_exhibit in original_data["exhibits"]:
        eid = orig_exhibit["id"]
        vlm_exhibit = next((e for e in r_scene_analysis["exhibits"] if e["id"] == eid), {})

        exhibit_enriched = {
            "id": eid,
            "name": vlm_exhibit.get("name", orig_exhibit["name"]),
            "type": orig_exhibit["type"],
            "description": vlm_exhibit.get("description", ""),
            "style": vlm_exhibit.get("style", ""),
            "colors": vlm_exhibit.get("colors", []),
            "location": vlm_exhibit.get("position", ""),
            "content_type": vlm_exhibit.get("content_type", ""),
            "material": vlm_exhibit.get("material", ""),
            "frame": vlm_exhibit.get("frame", ""),
            "bbox": orig_exhibit["bbox"],
            "center": orig_exhibit["center"],
            "area": orig_exhibit["area"],
            "sam_score": orig_exhibit["sam_score"],
            "gaze_count": orig_exhibit["gaze_count"],
            "total_duration": orig_exhibit["total_duration"],
            "avg_duration": orig_exhibit["avg_duration"],
            "attention_level": orig_exhibit["attention_level"],
            "attention_label": attention_map.get(orig_exhibit["attention_level"], ""),
            "sequence": vlm_exhibit.get("gaze_data", {}).get("sequence", 0)
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
        vlm_exhibit = next((e for e in r_scene_analysis["exhibits"] if e["id"] == eid), {})

        fixation_enriched = {
            "sequence": fixation["sequence"],
            "exhibit_id": fixation["exhibit_id"],
            "exhibit_name": fixation["exhibit_name"],
            "center": fixation["center"],
            "duration": fixation["duration"],
            "score": fixation["score"],
            "description": vlm_exhibit.get("description", ""),
            "style": vlm_exhibit.get("style", ""),
            "content_type": vlm_exhibit.get("content_type", "")
        }
        enriched_data["fixations"].append(fixation_enriched)

    # 保存文件
    output_path = base_dir / "data/outputs/R/scene_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    return enriched_data

if __name__ == "__main__":
    data = create_enriched_r_json()

    print("="*100)
    print(" R SCENE - ENRICHED ANALYSIS DATA")
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
    print("FIGURE CAPTION FOR SCENE R:")
    print("="*100)
    print("""
Figure X: Visual attention analysis of contemporary art gallery. (a) Original exhibition
scene (1100×600px) featuring 9 exhibits (8 paintings, 1 sculpture) arranged across left wall,
center pedestal, and right wall. (b) SAM2 segmentation results with color-coded region masks
showing detected exhibit boundaries with bounding boxes and center points. (c) Attention heatmap
with Gaussian kernel density estimation revealing spatial distribution with strongest attention
on upper-right wall (E8: 6.69s, E9: 6.60s). (d) Gaze trajectory (1→2→4→3→5→6→7→8→9)
demonstrating left-to-right scanning pattern with central sculpture serving as visual anchor.
Total viewing duration: 53.34s (mean: 5.93s ± 0.51s).
    """.strip())

    print("\nFile saved to: data/outputs/R/scene_analysis.json")
