#!/usr/bin/env python3
"""
生成学术风格的表格数据JSON
用于绘制类似论文中TABLE I/II格式的表格
"""
import json
from pathlib import Path

# R场景的VLM分析数据
vlm_data_r = {
    "E1": {
        "name": "Landscape Abstraction",
        "style": "Abstract/Minimalist",
        "description": "Green gradient with black tones, abstract natural landscape",
        "location": "Left Wall - Position 1"
    },
    "E2": {
        "name": "Portrait Study",
        "style": "Contemporary/Minimalist",
        "description": "Grey background with minimalist black human portrait",
        "location": "Left Wall - Position 2"
    },
    "E3": {
        "name": "Dark Composition",
        "style": "Abstract/Experimental",
        "description": "Dark abstract composition with black and dark grey tones",
        "location": "Left Wall - Position 3"
    },
    "E4": {
        "name": "Light Minimalist",
        "style": "Minimalist/Contemporary",
        "description": "Light-toned minimalist work with light grey and off-white",
        "location": "Left Wall - Position 4"
    },
    "E5": {
        "name": "Central Sculpture",
        "style": "Contemporary Sculpture",
        "description": "Abstract form on white cubic pedestal",
        "location": "Center - Pedestal"
    },
    "E6": {
        "name": "Sculpture Base",
        "style": "Contemporary",
        "description": "Sculpture pedestal and surrounding area",
        "location": "Center - Pedestal"
    },
    "E7": {
        "name": "Silhouette Landscape",
        "style": "Photographic/Minimalist",
        "description": "Blue sky gradient with black silhouette of person",
        "location": "Right Wall - Main"
    },
    "E8": {
        "name": "Upper Wall Art A",
        "style": "Contemporary",
        "description": "Upper wall display element",
        "location": "Upper Right Wall"
    },
    "E9": {
        "name": "Upper Wall Art B",
        "style": "Contemporary",
        "description": "Secondary upper wall display",
        "location": "Upper Right Wall"
    }
}

# G场景的VLM分析数据
vlm_data_g = {
    "E1": {
        "name": "Flower and Bird Painting",
        "style": "Chinese Ink Painting (Xieyi)",
        "description": "Bird and flower theme, expressing natural vitality through wash techniques",
        "location": "Left Wall - Position 1"
    },
    "E2": {
        "name": "Landscape Painting",
        "style": "Chinese Ink Landscape",
        "description": "Mountains, water and vegetation with layered ink showing spatial depth",
        "location": "Left Wall - Position 2"
    },
    "E3": {
        "name": "Architecture and Figure",
        "style": "Chinese Ink Painting (Gongbi)",
        "description": "Traditional Chinese architecture with figure, possibly historical scene",
        "location": "Center - Pedestal"
    },
    "E4": {
        "name": "Ink Landscape",
        "style": "Chinese Ink Landscape (Xieyi)",
        "description": "Black, grey and white tones depicting mountains and mist",
        "location": "Right Wall - Position 1"
    },
    "E5": {
        "name": "Bird and Flower Mood",
        "style": "Chinese Ink Painting (Xieyi)",
        "description": "Trees and flying birds in soft blue and yellow tones",
        "location": "Right Wall - Position 2"
    }
}

def generate_academic_table_data(original_json_path, vlm_data, output_path, scene_name):
    """生成学术风格的表格数据"""

    # 读取原始数据
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)

    # 注意力等级映射
    attention_map = {"A": "High", "B": "Medium-High", "C": "Medium", "D": "Medium-Low", "E": "Low", "-": "None"}

    # 构建表格数据
    table_data = {
        "scene": scene_name,
        "timestamp": original_data.get("timestamp", ""),
        "table_info": {
            "title": f"Exhibit Gaze Analysis - Scene {scene_name}",
            "columns": [
                {"id": "exhibit_id", "label": "ID", "width": 0.08},
                {"id": "name", "label": "Exhibit Name", "width": 0.25},
                {"id": "type", "label": "Type", "width": 0.12},
                {"id": "location", "label": "Location", "width": 0.20},
                {"id": "gaze_count", "label": "Gaze Count", "width": 0.10},
                {"id": "duration", "label": "Duration (s)", "width": 0.10},
                {"id": "attention", "label": "Attention Level", "width": 0.15}
            ]
        },
        "rows": [],
        "statistics": {}
    }

    # 计算统计数据
    durations = []
    gaze_counts = []
    attention_levels = []

    for exhibit in original_data["exhibits"]:
        eid = exhibit["id"]

        # 获取VLM增强的信息
        vlm_info = vlm_data.get(eid, {})
        enhanced_name = vlm_info.get("name", exhibit["name"])
        location = vlm_info.get("location", "")

        # 构建行数据
        row = {
            "exhibit_id": eid,
            "name": enhanced_name,
            "type": exhibit["type"],
            "location": location,
            "gaze_count": exhibit["gaze_count"],
            "duration": round(exhibit["total_duration"], 2),
            "attention": attention_map.get(exhibit["attention_level"], exhibit["attention_level"])
        }
        table_data["rows"].append(row)

        # 收集统计数据
        if exhibit["gaze_count"] > 0:
            durations.append(exhibit["total_duration"])
            gaze_counts.append(exhibit["gaze_count"])
        if exhibit["attention_level"] != "-":
            attention_levels.append(exhibit["attention_level"])

    # 计算统计汇总
    if durations:
        table_data["statistics"] = {
            "total_exhibits": len(original_data["exhibits"]),
            "gazed_exhibits": original_data["summary"]["gazed_exhibits"],
            "total_fixations": original_data["summary"]["total_fixations"],
            "mean_duration": round(sum(durations) / len(durations), 2),
            "total_duration": round(sum(durations), 2),
            "std_duration": round((sum((d - sum(durations)/len(durations))**2 for d in durations) / len(durations))**0.5, 2)
        }

    # 保存文件
    with open(output_path, 'w') as f:
        json.dump(table_data, f, indent=2, ensure_ascii=False)

    return table_data

def print_table_preview(table_data):
    """打印表格预览"""
    print("\n" + "="*100)
    print(f" {table_data['table_info']['title']} ")
    print("="*100)

    # 打印表头
    cols = table_data["table_info"]["columns"]
    header = " | ".join([f"{col['label']:<20}" for col in cols])
    print(header)
    print("-" * len(header))

    # 打印数据行
    for row in table_data["rows"]:
        row_str = " | ".join([
            f"{row['exhibit_id']:<20}",
            f"{row['name'][:20]:<20}",
            f"{row['type']:<20}",
            f"{row['location']:<20}",
            f"{row['gaze_count']:<20}",
            f"{row['duration']:<20}",
            f"{row['attention']:<20}"
        ])
        print(row_str)

    # 打印统计行
    if table_data["statistics"]:
        print("-" * len(header))
        stats = table_data["statistics"]
        print(f"Statistics: Total Exhibits: {stats['total_exhibits']} | "
              f"Gazed: {stats['gazed_exhibits']} | "
              f"Mean Duration: {stats['mean_duration']}s | "
              f"Total Duration: {stats['total_duration']}s")
    print("="*100)

if __name__ == "__main__":
    base_dir = Path("/home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026")

    # 处理R场景
    print("Processing Scene R...")
    table_r = generate_academic_table_data(
        base_dir / "data/outputs/R/table_data.json",
        vlm_data_r,
        base_dir / "data/outputs/R/table_academic.json",
        "R"
    )
    print_table_preview(table_r)

    # 处理G场景
    print("\nProcessing Scene G...")
    table_g = generate_academic_table_data(
        base_dir / "data/outputs/G/table_data.json",
        vlm_data_g,
        base_dir / "data/outputs/G/table_academic.json",
        "G"
    )
    print_table_preview(table_g)

    print("\nFiles generated:")
    print(f"  - {base_dir}/data/outputs/R/table_academic.json")
    print(f"  - {base_dir}/data/outputs/G/table_academic.json")
