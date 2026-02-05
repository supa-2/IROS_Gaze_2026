import pandas as pd
import numpy as np
import json
import os
import re
import glob
import argparse
from typing import List, Dict, Optional
from collections import Counter
from dataclasses import dataclass

# ==========================================
# 1. 基础配置
# ==========================================

DEFAULT_MAP_PATH = "skills/topology/assets/OS.xls"
DEFAULT_DATA_DIR = "data"
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

# ==========================================
# 2. 地图加载器
# ==========================================

class MapLoader:
    def __init__(self, map_path: str):
        self.items: Dict[str, Exhibit] = {}
        self._load_map(map_path)

    def _clean_text(self, text):
        if pd.isna(text): return ""
        s = str(text).strip().replace('\n', ' ').replace('\r', '')
        return re.sub(r'\s+', ' ', s)

    def _clean_id(self, eid):
        return str(eid).strip().upper().replace(" ", "")

    def _parse_neighbors(self, val) -> List[str]:
        if pd.isna(val): return []
        raw_list = re.split(r'[,;，；]', str(val))
        clean_list = []
        for item in raw_list:
            cid = self._clean_id(item)
            if len(cid) > 2: clean_list.append(cid)
        return clean_list

    def _load_map(self, path):
        print(f"[Map] 正在读取地图: {path}")
        try:
            if path.lower().endswith('.csv'): 
                try: df = pd.read_csv(path, encoding='utf-8', dtype={'ID': str})
                except: df = pd.read_csv(path, encoding='gbk', dtype={'ID': str})
            else: df = pd.read_excel(path, dtype={'ID': str})
        except Exception as e:
            print(f"[Error] {e}"); return

        for _, row in df.iterrows():
            eid = self._clean_id(row['ID'])
            if not eid.startswith('OS-'): continue
            
            name = self._clean_text(row.get('Name', '未命名展品'))
            desc = self._clean_text(row.get('Visual_Features', ''))
            if not desc or desc.lower() == 'nan':
                desc = f"这是一个{self._clean_text(row.get('Type', '展品'))}"
            
            neighbors = self._parse_neighbors(row.get('Neighbors (含关系)', ''))
            self.items[eid] = Exhibit(eid, name, desc, neighbors)
            
        print(f"[Map] 加载完成，包含 {len(self.items)} 个节点")

    def get_context_items(self, center_id: str, extra_id: Optional[str] = None) -> List[Exhibit]:
        if center_id not in self.items: return []
        center = self.items[center_id]
        ctx_map = {center_id: center}
        for nid in center.neighbors:
            if nid in self.items: ctx_map[nid] = self.items[nid]
        if extra_id and extra_id in self.items:
            ctx_map[extra_id] = self.items[extra_id]
        return list(ctx_map.values())

# ==========================================
# 3. 数据生成器 (V16: Scientific Rigor)
# ==========================================

class DatasetGenerator:
    def __init__(self, map_loader: MapLoader):
        self.map = map_loader

    def parse_time(self, t_str):
        try:
            parts = str(t_str).strip().split(':')
            if len(parts) == 3:
                return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        except: return 0.0
        return 0.0
    
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
        except: return []

        if 'Hot key' not in df.columns: return []
        
        df['Hot key'] = df['Hot key'].fillna('')
        df = df[df['Hot key'].str.contains('OS')].copy()
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

        if not raw_fixs: return []

        # Log Normalization
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
        
        # 任务 1: 下一步预测
        for i in range(len(raw_fixs) - 1):
            curr = raw_fixs[i]
            nxt = raw_fixs[i+1]
            ctx = self.map.get_context_items(curr.id, nxt.id)
            samples.append(self._create_prediction_sample(ctx, [curr], nxt))
            STATS["tasks"]["prediction"] += 1

        # 任务 2: 路径规划
        for i in range(len(raw_fixs) - 1):
            seq = raw_fixs[i : min(len(raw_fixs), i+5)]
            if len(seq) < 2: continue
            
            ctx_items = self.map.get_context_items(seq[0].id)
            existing_ids = set(x.id for x in ctx_items)
            for p in seq[1:]:
                if p.id not in existing_ids and p.id in self.map.items:
                    ctx_items.append(self.map.items[p.id])
                    existing_ids.add(p.id)
            
            samples.append(self._create_planning_sample(ctx_items, seq))
            STATS["tasks"]["planning"] += 1

        # 任务 3: 显著性归因 (原 Task C 修改版)
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
        """
        V16 修改：显著性归因
        不再说"用户因为...所以看了...", 而是说"高关注度对应了...特征"
        """
        ipt = self._format_input(items)
        return {
            # 措辞更加客观，学术上更安全
            "instruction": f"识别并提取出与目标展品【{target.name}】的高强度关注（{target.grade}级）相对应的视觉显著性特征。",
            "input": ipt,
            "output": f"特征归因: 目标【{target.name}】被标记为 {target.grade} 级热点。其对应的视觉显著性特征包括：{target.description}。这些特征在当前场景中具有较高的视觉权重。"
        }

# ==========================================
# 4. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DEFAULT_DATA_DIR)
    parser.add_argument('--map', default=DEFAULT_MAP_PATH)
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    
    if not os.path.exists(args.output): os.makedirs(args.output)
    
    map_path = args.map
    if not os.path.exists(map_path) and os.path.exists(os.path.basename(map_path)):
        map_path = os.path.basename(map_path)
    map_loader = MapLoader(map_path)
    
    all_files = set(glob.glob(os.path.join(args.data, "**", "*.csv"), recursive=True) + 
                   glob.glob(os.path.join(args.data, "**", "*.xlsx"), recursive=True))
    raw_files = [f for f in all_files if "processed" not in f and ("OS" in f or "raw" in f.lower()) and not os.path.basename(f).startswith("~$")]
    
    print(f"开始 V16 学术严谨版处理 ({len(raw_files)} 文件)...")
    
    gen = DatasetGenerator(map_loader)
    data = []
    for f in raw_files:
        data.extend(gen.process_file(f))
        
    out_path = os.path.join(args.output, "gaze_finetune_v16_scientific.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*60)
    print(f"🚀 总样本数: {len(data)}")
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