import pandas as pd

# 检查原始数据格式
df = pd.read_excel('data/OS/chenzhuangjin/raw_Chenzhuangjin_250923102755_260126_161616.xlsx')

print('列名:', df.columns.tolist())
print('\n前20行数据:')
print(df.head(20))
print('\nHot key的统计信息:')
print(f'总记录数: {len(df)}')
print(f'有Hot key的记录数: {df["Hot key"].notna().sum()}')
print(f'Hot key唯一值数量: {df["Hot key"].nunique()}')

print('\nHot key的取值:')
print(df['Hot key'].value_counts().head(10))

print('\nHot key对应的Video Time样本:')
hot_key_records = df[df['Hot key'].notna()].head(10)
for _, row in hot_key_records.iterrows():
    print(f"Hot key: {row['Hot key']:10} | Video Time: {row['Video Time[HH:mm:ss.ms]']}")
