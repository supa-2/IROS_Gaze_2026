import pandas as pd

df = pd.read_excel('skills/topology/assets/OS.xls')

print('Columns:', df.columns.tolist())
print('\nFirst 3 rows:')
print(df.head(3))

if 'Neighbors (含关系)' in df.columns:
    print('\nNeighbors example:', df.iloc[0]['Neighbors (含关系)'])
else:
    print('\nNo neighbors column found')
