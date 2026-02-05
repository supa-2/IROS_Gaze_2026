import pandas as pd
import numpy as np
import json
import os
import glob
import argparse
from typing import List, Dict, Optional
from collections import Counter
from dataclasses import dataclass

DEFAULT_MAP_PATH = "skills/topology/assets/TH.xlsx"
DEFAULT_DATA_DIR = "data/TH"
DEFAULT_OUTPUT_DIR = "data/processed"

STATS = {
    "total_raw": 0,
    "final_samples": 0,
    "tasks": Counter()
}

@dataclass
class Exhibit:
    id: str
    name: str
    description: str
    neighbors: List[str]

@dataclass
class Fixation:
    id: str
    name: str
    raw_duration: float
    description: str = "" 
    grade: str = "E"
    
    @property
    def output_str(self):
        return f"[{self.grade}] {self.name}"

class MapLoader:
    def __init__(self, map_path: str):
        self.items = {}
        self.load_map(map_path)
    
    def load_map(self, map_path: str):
        df = pd.read_excel(map_path)
        
        for _, row in df.iterrows():
            eid = str(row['ID']).strip()
            name = str(row['Name']).strip()
            desc = str(row['Visual_Features']).strip()
            neighbors_str = str(row['Neighbors (含关系)']).strip()
            
            neighbors = []
            if neighbors_str and neighbors_str != 'nan':
                neighbors_str = neighbors_str.replace('nan', '').strip()
                if neighbors_str:
                    neighbors.append(neighbors_str)
            
            self.items[eid] = Exhibit(eid, name, desc, neighbors)
    
    def get_context_items(self, current_id: str, next_id: str = None) -> List[Exhibit]:
        ctx = []
        
        if current_id in self.items:
            ctx.append(self.items[current_id])
            
            current_item = self.items[current_id]
            for neighbor_id in current_item.neighbors:
                if neighbor_id in self.items:
                    ctx.append(self.items[neighbor_id])
        
        if next_id and next_id in self.items and next_id not in [x.id for x in ctx]:
            ctx.append(self.items[next_id])
        
        return ctx

class DatasetGenerator:
    def __init__(self, map_loader: MapLoader):
        self.map = map_loader
    
    def parse_time(self, time_str):
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                h, m, s = parts
                s_parts = s.split('.')
                seconds = int(h) * 3600 + int(m) * 60 + float(s)
                return seconds
        except:
            return 0
        return 0
    
    def _clean_id(self, eid):
        return str(eid).strip().upper().replace(" ", "")
    
    def _assign_grade(self, score: float) -> str:
        if score >= 0.8: return "A"
        if score >= 0.6: return "B"
        if score >= 0.4: return "C"
        if score >= 0.2: return "D"
        return "E"
    
    def process_file(self, file_path: str) -> List[Dict]:
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path, dtype={'Hot key': str}, low_memory=False)
            else:
                df = pd.read_excel(file_path, dtype={'Hot key': str})
        except: 
            return []
        
        if 'Hot key' not in df.columns: 
            return []
        
        df['Hot key'] = df['Hot key'].fillna('')
        df = df[df['Hot key'].str.contains('TH')].copy()
        df['ts'] = df['Video Time[HH:mm:ss.ms]'].apply(self.parse_time)
        df = df.sort_values('ts')
        
        raw_fixs = []
        curr_id, start_t, last_t = None, 0, 0
        
        for _, row in df.iterrows():
            eid = self._clean_id(row['Hot key'])
            t = row['ts']
            
            if eid != curr_id:
                if curr_id:
                    dur = max(0.01, last_t - start_t)
                    if curr_id in self.map.items:
                        item = self.map.items[curr_id]
                        raw_fixs.append(Fixation(curr_id, item.name, dur, item.description))
                curr_id, start_t = eid, t
            last_t = t
        
        if curr_id and curr_id in self.map.items:
            dur = max(0.01, last_t - start_t)
            item = self.map.items[curr_id]
            raw_fixs.append(Fixation(curr_id, item.name, dur, item.description))
        
        if not raw_fixs: 
            return []
        
        durations = np.array([f.raw_duration for f in raw_fixs])
        log_durations = np.log(durations + 1e-5)
        min_log = np.min(log_durations)
        max_log = np.max(log_durations)
        range_log = max_log - min_log if (max_log - min_log) > 0 else 1.0
        
        for i, f in enumerate(raw_fixs):
            score = (log_durations[i] - min_log) / range_log
            f.grade = self._assign_grade(score)
        
        STATS["total_raw"] += len(raw_fixs)
        
        samples = []
        
        for i in range(len(raw_fixs) - 1):
            curr = raw_fixs[i]
            nxt = raw_fixs[i+1]
            ctx = self.map.get_context_items(curr.id, nxt.id)
            samples.append(self._create_prediction_sample(ctx, [curr], nxt))
            STATS["tasks"]["prediction"] += 1
        
        for i in range(len(raw_fixs) - 1):
            seq = raw_fixs[i : min(len(raw_fixs), i+5)]
            if len(seq) < 2: 
                continue
            
            ctx_items = self.map.get_context_items(seq[0].id)
            existing_ids = set(x.id for x in ctx_items)
            for p in seq[1:]:
                if p.id not in existing_ids and p.id in self.map.items:
                    ctx_items.append(self.map.items[p.id])
                    existing_ids.add(p.id)
            
            samples.append(self._create_planning_sample(ctx_items, seq))
            STATS["tasks"]["planning"] += 1
        
        for f in raw_fixs:
            if f.grade in ['A', 'B']:
                ctx_items = self.map.get_context_items(f.id)
                samples.append(self._create_attribution_sample(ctx_items, f))
                STATS["tasks"]["reasoning"] += 1
        
        STATS["final_samples"] += len(samples)
        return samples
    
    def _format_input(self, items: List[Exhibit]) -> str:
        unique = {x.id: x for x in items}.values()
        lines = ["当前场景可见展品(Visual Context):"]
        for x in unique:
            desc = x.description[:100]
            lines.append(f"- 【{x.name}】: {desc}")
        return "\n".join(lines)
    
    def _create_prediction_sample(self, items, hist, nxt):
        ipt = self._format_input(items)
        ht = "\n".join([f"{i+1}. {h.output_str}" for i,h in enumerate(hist)])
        return {
            "instruction": "基于当前视觉场景和历史注视行为，预测用户下一个关注目标的注意力等级 (A/B/C/D/E)。",
            "input": f"{ipt}\n\n历史行为:\n{ht}",
            "output": f"下一步预测: {nxt.output_str}"
        }
    
    def _create_planning_sample(self, items, seq):
        ipt = self._format_input(items)
        ot = "\n".join([f"{i+1}. {p.output_str}" for i,p in enumerate(seq)])
        return {
            "instruction": "规划一条包含注意力等级(A-E)的视觉扫描路径，用于构建注意力热图。",
            "input": ipt,
            "output": f"热度路径规划:\n{ot}"
        }
    
    def _create_attribution_sample(self, items, target):
        ipt = self._format_input(items)
        return {
            "instruction": f"识别并提取出与目标展品【{target.name}】的高强度关注（{target.grade}级）相对应的视觉显著性特征。",
            "input": ipt,
            "output": f"特征归因: 目标【{target.name}】被标记为 {target.grade} 级热点。其对应的视觉显著性特征包括：{target.description}。这些特征在当前场景中具有较高的视觉权重。"
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DEFAULT_DATA_DIR)
    parser.add_argument('--map', default=DEFAULT_MAP_PATH)
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    
    if not os.path.exists(args.output): 
        os.makedirs(args.output)
    
    map_path = args.map
    if not os.path.exists(map_path) and os.path.exists(os.path.basename(map_path)):
        map_path = os.path.basename(map_path)
    map_loader = MapLoader(map_path)
    
    all_files = set(glob.glob(os.path.join(args.data, "**", "*.csv"), recursive=True) + 
                   glob.glob(os.path.join(args.data, "**", "*.xlsx"), recursive=True))
    raw_files = [f for f in all_files if "processed" not in f and ("TH" in f or "raw" in f.lower()) and not os.path.basename(f).startswith("~$")]
    
    print(f"开始 TH 多任务数据集处理...")
    print(f"找到的文件总数: {len(raw_files)}")
    for i, f in enumerate(raw_files, 1):
        print(f"  {i}. {os.path.basename(f)}")
    
    gen = DatasetGenerator(map_loader)
    data = []
    processed_count = 0
    for f in raw_files:
        print(f"\n处理文件: {os.path.basename(f)}")
        samples = gen.process_file(f)
        if samples:
            data.extend(samples)
            processed_count += 1
            print(f"  生成样本数: {len(samples)}")
        else:
            print(f"  无有效数据")
    
    out_path = os.path.join(args.output, "th_gaze_finetune_multitask.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print(f"🚀 总样本数: {len(data)}")
    print(f"📁 成功处理文件数: {processed_count}/{len(raw_files)}")
    print("-" * 30)
    print("【任务分布】")
    print(f"1. 下一步预测: {STATS['tasks']['prediction']} 条")
    print(f"2. 路径规划:   {STATS['tasks']['planning']} 条")
    print(f"3. 显著性归因: {STATS['tasks']['reasoning']} 条")
    print("-" * 30)
    
    if data:
        reasoning_samples = [s for s in data if "特征归因" in s['output']]
        if reasoning_samples:
            print("[显著性归因样本示例]")
            print(f"Instruction: {reasoning_samples[0]['instruction']}")
            print(f"Output: {reasoning_samples[0]['output']}")
    print("="*60)

if __name__ == "__main__":
    main()
