import pandas as pd
import re

df = pd.read_excel('skills/topology/assets/OS.xls')

print('Sample exhibits and their neighbors:')
print('-' * 80)

# 创建邻接字典
adjacency = {}
for _, row in df.iterrows():
    exhibit_id = str(row['ID']).strip()
    neighbors_raw = str(row['Neighbors (含关系)'])
    neighbors = []
    if neighbors_raw and neighbors_raw != 'nan':
        raw_list = re.split(r'[;，,]', neighbors_raw)
        for item in raw_list:
            item = item.strip()
            if item and item != 'nan':
                neighbors.append(item)
    adjacency[exhibit_id] = neighbors

# 打印一些样例
for exhibit_id in ['OS-AO1', 'OS-AO4', 'OS-AO7', 'OS-AO10']:
    if exhibit_id in adjacency:
        print(f'{exhibit_id:12} -> {adjacency[exhibit_id]}')
        # 反向查找前驱
        predecessors = [e for e, ns in adjacency.items() if exhibit_id in ns]
        print(f'          <- {predecessors}')
        print()

print('-' * 80)
print('Total exhibits:', len(df))
