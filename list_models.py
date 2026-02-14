"""
列出 Replicate 上可用的模型
"""

import sys
import os

# 设置 UTF-8 输出编码 (Windows 兼容)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import replicate
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取 API token
api_token = os.getenv("REPLICATE_API_TOKEN")
if not api_token:
    print("REPLICATE_API_TOKEN not set")
    sys.exit(1)

print("Connecting to Replicate...")
print(f"API Token: {api_token[:20]}...")
print()

# 初始化客户端
client = replicate.Client(api_token=api_token)

try:
    print("Listing all models...")
    models = client.models.list()
    print(f"Total models: {len(models)}")

    # 查找分割相关的模型
    seg_models = [m for m in models if 'segment' in m.name.lower()]
    print(f"\nFound {len(seg_models)} segmentation models:")
    for m in seg_models[:15]:
        print(f"  - {m.owner}/{m.name}")

    # 显示所有模型
    print(f"\nAll models:")
    for i, m in enumerate(models[:25]):
        print(f"{i+1}. {m.owner}/{m.name}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
