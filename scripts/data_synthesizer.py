import sys
import os
import random
import json
import networkx as nx

# 引用 skills 里的代码
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'skills', 'topology'))
from graph_engine import TopologyEngine

def generate_random_walks(map_name='TH', num_walkers=10, min_steps=5, max_steps=10):
    """
    模拟游客在展馆内的随机游走，生成测试数据。
    """
    engine = TopologyEngine(map_name)
    graph = engine.graph
    
    # 获取所有节点作为可能的起点
    all_nodes = list(graph.nodes())
    dataset = []

    print(f"开始生成 {map_name} 地图的虚拟轨迹...")

    for i in range(num_walkers):
        # 随机选一个起点
        current_node = random.choice(all_nodes)
        path = [current_node]
        
        # 随机走 N 步
        steps = random.randint(min_steps, max_steps)
        for _ in range(steps):
            # 获取邻居
            neighbors = list(graph.neighbors(current_node))
            if not neighbors:
                break # 走到死胡同了
            
            # 简单策略：随机选一个邻居走下去 (模拟普通游客)
            next_node = random.choice(neighbors)
            path.append(next_node)
            current_node = next_node
            
        dataset.append({
            "walker_id": i,
            "path": path
        })
    
    return dataset

if __name__ == "__main__":
    # 生成 5 条轨迹用于测试
    data = generate_random_walks(map_name='TH', num_walkers=5)
    
    # 保存为 JSON
    output_path = os.path.join(os.path.dirname(__file__), 'synthetic_test_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"生成完毕！数据已保存至: {output_path}")
    print(f"示例轨迹: {data[0]['path']}")