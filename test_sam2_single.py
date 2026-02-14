"""
SAM 2 单次测试 - 测试 lucataco/segment-anything-2
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

# 测试图片
TEST_IMAGE = "data/test_museum.jpg"

# 获取 API token
api_token = os.getenv("REPLICATE_API_TOKEN")
if not api_token:
    print("❌ REPLICATE_API_TOKEN 未设置")
    sys.exit(1)

print("=" * 60)
print("  SAM 2 单次测试")
print("=" * 60)
print(f"✅ API Token: {api_token[:20]}...")
print(f"✅ 测试图片: {TEST_IMAGE}")
print()

# 检查测试图片是否存在
if not os.path.exists(TEST_IMAGE):
    print(f"❌ 测试图片不存在: {TEST_IMAGE}")
    sys.exit(1)

# 初始化 Replicate 客户端
client = replicate.Client(api_token=api_token)

# 使用 cjwbw/semantic-segment-anything 模型
model_version = "cjwbw/semantic-segment-anything"

print(f"🔍 测试模型: {model_version}")
print()

# 读取测试图片
image_file = open(TEST_IMAGE, "rb")

try:
    print("⏳ 调用 Replicate API...")
    print("   (这可能需要 10-30 秒)...")

    output = client.run(
        model_version,
        input={
            "image": image_file
        }
    )

    print("✅ 调用成功!")
    print()
    print("=" * 60)
    print("  输出结果")
    print("=" * 60)

    # 打印输出结构
    print(f"   输出类型: {type(output)}")

    if isinstance(output, dict):
        print(f"   键: {list(output.keys())}")
        if 'masks' in output:
            print(f"   masks 数量: {len(output['masks'])}")
            print(f"   masks 类型: {type(output['masks'][0]) if output['masks'] else 'N/A'}")
        if 'scores' in output:
            print(f"   scores 数量: {len(output['scores'])}")
        if 'annotated_image' in output:
            print(f"   annotated_image: {len(output['annotated_image'])} bytes")
    elif isinstance(output, list):
        print(f"   列表长度: {len(output)}")

    # 保存输出
    import json
    output_path = "data/outputs/sam2_result.json"
    os.makedirs("data/outputs", exist_ok=True)

    # 尝试序列化输出
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if isinstance(output, dict):
                json.dump(output, f, ensure_ascii=False, indent=2, default=str)
            else:
                json.dump({"output": str(output)}, f, ensure_ascii=False, indent=2)
        print(f"✅ 结果已保存到: {output_path}")
    except Exception as e:
        print(f"⚠️  无法序列化输出: {e}")
        print(f"   输出: {output}")

except replicate.exceptions.ReplicateError as e:
    print(f"❌ Replicate API 错误:")
    print(f"   状态码: {e.status}")
    print(f"   详情: {e.detail}")
except Exception as e:
    print(f"❌ 未知错误: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
