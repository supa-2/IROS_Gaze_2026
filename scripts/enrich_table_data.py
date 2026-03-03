#!/usr/bin/env python3
"""
使用VLM分析图片并生成详细的展品信息JSON
"""
import json
import os
from pathlib import Path

# VLM分析结果 - R场景
vlm_analysis_r = {
    "exhibits": [
        {
            "id": "E1",
            "name": "Landscape Abstraction",
            "type": "Painting",
            "location": "Left Wall - Position 1",
            "description": "Green gradient with black tones, abstract natural landscape (grassland or water)",
            "style": "Abstract/Minimalist",
            "colors": ["Green", "Black", "Dark Green"],
            "frame": "Black frame",
            "content_type": "Abstract Landscape",
            "position_detail": "Lower left wall, leftmost position"
        },
        {
            "id": "E2",
            "name": "Portrait Study",
            "type": "Painting",
            "location": "Left Wall - Position 2",
            "description": "Grey background with a minimalist black human portrait (short hair, simple facial details)",
            "style": "Contemporary/Minimalist",
            "colors": ["Grey", "Black", "White"],
            "frame": "Black frame",
            "content_type": "Portrait",
            "position_detail": "Left wall, second position from left"
        },
        {
            "id": "E3",
            "name": "Dark Composition",
            "type": "Painting",
            "location": "Left Wall - Position 3",
            "description": "Dark abstract composition with black and dark grey tones, possibly lines or geometric shapes",
            "style": "Abstract/Experimental",
            "colors": ["Black", "Dark Grey", "Charcoal"],
            "frame": "Black frame",
            "content_type": "Abstract",
            "position_detail": "Left wall, third position"
        },
        {
            "id": "E4",
            "name": "Light Minimalist",
            "type": "Painting",
            "location": "Left Wall - Position 4",
            "description": "Light-toned minimalist work with light grey and off-white colors",
            "style": "Minimalist/Contemporary",
            "colors": ["Light Grey", "Off-White", "Beige"],
            "frame": "Black frame",
            "content_type": "Abstract Minimalist",
            "position_detail": "Left wall, fourth position"
        },
        {
            "id": "E5",
            "name": "Central Sculpture",
            "type": "Sculpture",
            "location": "Center - Pedestal",
            "description": "Abstract form on white cubic pedestal, organic/geometric hybrid shape with curves",
            "style": "Contemporary Sculpture",
            "colors": ["Grey", "Stone"],
            "material": "Ceramic or Stone",
            "content_type": "Abstract Sculpture",
            "position_detail": "Center of room on white pedestal"
        },
        {
            "id": "E6",
            "name": "Sculpture Base Area",
            "type": "Sculpture Element",
            "location": "Center - Pedestal",
            "description": "Sculpture pedestal and surrounding area",
            "style": "Contemporary",
            "colors": ["White", "Grey"],
            "material": "White painted wood/pedestal",
            "content_type": "Display Element",
            "position_detail": "Central display area"
        },
        {
            "id": "E7",
            "name": "Silhouette Landscape",
            "type": "Painting",
            "location": "Right Wall - Main",
            "description": "Blue sky gradient (light to dark blue) with black silhouette of person standing in grass with vegetation",
            "style": "Photographic/Minimalist",
            "colors": ["Blue", "Black", "Light Blue"],
            "frame": "Black frame",
            "content_type": "Landscape with Figure",
            "position_detail": "Right wall, large format"
        },
        {
            "id": "E8",
            "name": "Upper Wall Element",
            "type": "Painting",
            "location": "Upper Right Wall",
            "description": "Upper wall display element",
            "style": "Contemporary",
            "colors": ["Mixed"],
            "frame": "Black frame",
            "content_type": "Wall Art",
            "position_detail": "Upper right section"
        },
        {
            "id": "E9",
            "name": "Secondary Upper Element",
            "type": "Painting",
            "location": "Upper Right Wall",
            "description": "Secondary upper wall display",
            "style": "Contemporary",
            "colors": ["Mixed"],
            "frame": "Black frame",
            "content_type": "Wall Art",
            "position_detail": "Upper right section, adjacent to E8"
        }
    ],
    "environment": {
        "wall_color": "White",
        "floor_color": "Grey",
        "lighting": "Track lighting from ceiling",
        "overall_style": "Modern minimalist gallery"
    }
}

def enrich_table_data(original_json_path, vlm_data, output_path):
    """将VLM数据整合到原始table_data.json中"""

    # 读取原始数据
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)

    # 创建exhibit映射
    exhibit_details = {e["id"]: e for e in vlm_data["exhibits"]}

    # 增强exhibits数据
    for exhibit in original_data["exhibits"]:
        eid = exhibit["id"]
        if eid in exhibit_details:
            details = exhibit_details[eid]
            exhibit["description"] = details.get("description", exhibit.get("description", ""))
            exhibit["style"] = details.get("style", "")
            exhibit["colors"] = details.get("colors", [])
            exhibit["location"] = details.get("location", "")
            exhibit["content_type"] = details.get("content_type", "")
            exhibit["material"] = details.get("material", "")
            exhibit["frame"] = details.get("frame", "")

    # 增强fixations数据
    for fixation in original_data["fixations"]:
        eid = fixation["exhibit_id"]
        if eid in exhibit_details:
            details = exhibit_details[eid]
            fixation["description"] = details.get("description", "")
            fixation["style"] = details.get("style", "")
            fixation["content_type"] = details.get("content_type", "")

    # 添加环境信息
    original_data["environment"] = vlm_data["environment"]

    # 添加用于表格的数据
    original_data["table_data"] = {
        "columns": ["ID", "Name", "Type", "Style", "Colors", "Location", "Content", "Gaze Count", "Duration (s)", "Attention"],
        "rows": []
    }

    attention_map = {"A": "High", "B": "Medium-High", "C": "Medium", "D": "Medium-Low", "E": "Low"}

    for exhibit in original_data["exhibits"]:
        original_data["table_data"]["rows"].append({
            "ID": exhibit["id"],
            "Name": exhibit["name"],
            "Type": exhibit["type"],
            "Style": exhibit.get("style", ""),
            "Colors": ", ".join(exhibit.get("colors", [])),
            "Location": exhibit.get("location", ""),
            "Content": exhibit.get("content_type", ""),
            "Gaze Count": exhibit["gaze_count"],
            "Duration (s)": round(exhibit["total_duration"], 2),
            "Attention": attention_map.get(exhibit["attention_level"], exhibit["attention_level"])
        })

    # 保存增强后的数据
    with open(output_path, 'w') as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)

    print(f"Enhanced data saved to: {output_path}")
    return original_data

if __name__ == "__main__":
    base_dir = Path("/home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026")

    # 处理R场景
    original_r = base_dir / "data/outputs/R/table_data.json"
    output_r = base_dir / "data/outputs/R/table_data_detailed.json"

    if original_r.exists():
        enriched_r = enrich_table_data(original_r, vlm_analysis_r, output_r)

        # 打印表格数据预览
        print("\n" + "="*60)
        print("TABLE DATA PREVIEW (R Scene)")
        print("="*60)
        print(f"{'ID':<6} {'Name':<25} {'Type':<12} {'Style':<20} {'Duration':<10} {'Attention':<12}")
        print("-"*100)
        for row in enriched_r["table_data"]["rows"]:
            print(f"{row['ID']:<6} {row['Name']:<25} {row['Type']:<12} {row['Style']:<20} {row['Duration (s)']:<10} {row['Attention']:<12}")
