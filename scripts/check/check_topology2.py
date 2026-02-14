import pandas as pd

df = pd.read_excel('skills/topology/assets/OS.xls')

print('Columns:', df.columns.tolist())
print('\nFirst 10 rows - ID and Neighbors:')
for i in range(min(10, len(df))):
    exhibit_id = df.iloc[i]['ID']
    neighbors = df.iloc[i]['Neighbors (含关系)']
    print(f'{i+1}. {exhibit_id:<15} -> {neighbors}')

print('\nTotal exhibits:', len(df))
