import pandas as pd

# 检查TH.xlsx地图数据集
df = pd.read_excel('skills/topology/assets/TH.xlsx')

print('列名:', df.columns.tolist())
print('\n前10行数据:')
print(df.head(10))
print('\n统计信息:')
print(f'总记录数: {len(df)}')
print(f'ID唯一值数量: {df["ID"].nunique()}')
