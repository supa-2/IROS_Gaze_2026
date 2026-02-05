import argparse
import pandas as pd
import networkx as nx
import re
import json
import os
import sys

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

class TopologyEngine:
    def __init__(self, map_name):
        self.graph = nx.DiGraph()
        self.map_name = map_name
        self._load_data(map_name)

    def _load_data(self, map_name):
        # 1. 自动寻找文件
        target_file = None
        for ext in ['.csv', '.xlsx', '.xls']:
            path = os.path.join(ASSETS_DIR, f"{map_name}{ext}")
            if os.path.exists(path):
                target_file = path
                break
        
        if not target_file:
            print(json.dumps({"error": f"Map file '{map_name}' not found."}))
            sys.exit(1)

        # 2. 读取数据 (智能适配表头)
        try:
            # 针对 TH 这种可能缺表头的文件做特殊处理
            # 简单的启发式：看第一行是不是包含 'ID'，如果不包含，假设它是无表头的
            preview = pd.read_excel(target_file, nrows=1) if target_file.endswith(('.xls', '.xlsx')) else pd.read_csv(target_file, nrows=1)
            
            if 'ID' in preview.columns:
                df = pd.read_excel(target_file) if target_file.endswith(('.xls', '.xlsx')) else pd.read_csv(target_file)
            else:
                # 强制补全表头
                header_names = ['ID', 'Name', 'Type', 'Visual_Features', 'Neighbors (含关系)', 'Image']
                if target_file.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(target_file, header=None, names=header_names)
                else:
                    df = pd.read_csv(target_file, header=None, names=header_names)
        except Exception as e:
            print(json.dumps({"error": f"File read error: {str(e)}"}))
            sys.exit(1)

        # 3. 构建图
        for _, row in df.iterrows():
            if pd.isna(row.get('ID')): continue
            curr_id = str(row['ID']).strip()
            
            self.graph.add_node(curr_id, 
                               name=row.get('Name', 'Unknown'),
                               type=row.get('Type', 'General'),
                               features=row.get('Visual_Features', ''))
            
            # --- 核心升级：通用正则解析 ---
            neighbors_raw = str(row.get('Neighbors (含关系)', ''))
            if neighbors_raw and neighbors_raw != 'nan':
                # 分割多个邻居 (支持分号或逗号)
                raw_list = re.split(r'[;，,]', neighbors_raw)
                for item in raw_list:
                    item = item.strip()
                    if not item: continue
                    
                    # 尝试匹配 "ID(关系)"
                    match_complex = re.match(r'([\w-]+)\((.*?)\)', item)
                    # 尝试匹配 纯 "ID"
                    match_simple = re.match(r'^([\w-]+)$', item)
                    
                    if match_complex:
                        target_id = match_complex.group(1)
                        relation = match_complex.group(2)
                        self.graph.add_edge(curr_id, target_id, relation=relation)
                    elif match_simple:
                        target_id = match_simple.group(1)
                        self.graph.add_edge(curr_id, target_id, relation="next") # 默认为 next

    def _trace_path(self, start_node, direction='successors', depth=3):
        """内部工具：顺藤摸瓜找路径"""
        path = []
        curr = start_node
        for _ in range(depth):
            if direction == 'successors':
                neighbors = list(self.graph.successors(curr))
            else:
                neighbors = list(self.graph.predecessors(curr))
            
            if not neighbors: break
            
            # 简单策略：如果有多个分支，取第一个 (后续可改为取 flow 关系)
            next_node = neighbors[0]
            
            # 获取信息
            info = self.graph.nodes[next_node]
            path.append({
                "id": next_node,
                "name": info.get('name'),
                "dist": len(path) + 1
            })
            curr = next_node
        return path

    def query_node(self, node_id):
        if node_id not in self.graph:
             return {"error": f"Node {node_id} not found."}

        node_info = self.graph.nodes[node_id]
        
        # --- 核心升级：获取前后文 ---
        # 1. 直接邻居 (Outbound)
        direct_connected = []
        for n in self.graph.successors(node_id):
            direct_connected.append({
                "id": n, 
                "relation": self.graph[node_id][n]['relation']
            })

        # 2. 链式上下文 (Previous 3 / Next 3)
        prev_chain = self._trace_path(node_id, 'predecessors', 3)
        next_chain = self._trace_path(node_id, 'successors', 3)

        return {
            "map": self.map_name,
            "id": node_id,
            "info": node_info,
            "context": {
                "previous_path": prev_chain,  # 你的"前3个"
                "next_path": next_chain,      # 你的"后3个"
                "direct_choices": direct_connected
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('--id')
    parser.add_argument('--from_id')
    parser.add_argument('--map', default='OS')
    args = parser.parse_args()
    
    engine = TopologyEngine(args.map)
    
    if args.action == 'query':
        if not args.id:
            print(json.dumps({"error": "Missing --id"}))
        else:
            print(json.dumps(engine.query_node(args.id), ensure_ascii=False))
            
    elif args.action == 'predict':
        # 预测逻辑暂时复用 query，IROS 论文中这里会接 LLM
        target = args.from_id if args.from_id else args.id
        print(json.dumps(engine.query_node(target), ensure_ascii=False))