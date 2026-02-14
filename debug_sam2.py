"""
SAM 2 调试脚本 - 测试不同的模型版本
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

# 测试图片（使用已创建的测试图片）
TEST_IMAGE = "data/test_museum.jpg"

# 可能的模型版本名称
MODEL_VERSIONS = [
    "meta/sam2-video",
    "lucataco/segment-anything-2",
]

# 获取 API token
api_token = os.getenv("REPLICATE_API_TOKEN")
if not api_token:
    print("❌ REPLICATE_API_TOKEN 未设置")
    sys.exit(1)

print("=" * 60)
print("  SAM 2 模型版本调试")
print("=" * 60)
print(f"✅ API Token: {api_token[:20]}...")
print(f"✅ 测试图片: {TEST_IMAGE}")
print()

# 检查测试图片是否存在
if not os.path.exists(TEST_IMAGE):
    print(f"❌ 测试图片不存在: {TEST_IMAGE}")
    # 创建测试图片
    print("📸 创建测试图片...")
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (800, 600), color='#f5f5f5')
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 150, 250, 350], fill='#e8dcc3', outline='#8b4513', width=3)
    draw.text((120, 170), 'Exhibit A', fill='black')
    draw.rectangle([400, 100, 550, 250], fill='#ffeaa7', outline='#ff6b6b', width=3)
    draw.text((420, 150), 'Exhibit B', fill='black')
    img.save(TEST_IMAGE, quality=95)
    print(f"✅ 测试图片已创建: {TEST_IMAGE}")

print()

# 读取测试图片
# Replicate API 需要使用文件对象，不是 bytes
image_file = open(TEST_IMAGE, "rb")

print(f"✅ 测试图片: {TEST_IMAGE}")
print()

# 初始化 Replicate 客户端
client = replicate.Client(api_token=api_token)

# 测试每个模型版本
for i, model_version in enumerate(MODEL_VERSIONS, 1):
    print("-" * 60)
    print(f"🔍 测试 {i}/{len(MODEL_VERSIONS)}: {model_version}")
    print("-" * 60)

    try:
        # 尝试调用模型
        print(f"⏳ 调用 Replicate API...")

        output = client.run(
            model_version,
            input={
                "image": image_file,
                "mask_prompt": "auto"
            }
        )

        print(f"✅ 成功! 模型: {model_version}")
        print(f"   输出类型: {type(output)}")

        # 打印输出结构
        if isinstance(output, dict):
            print(f"   键: {list(output.keys())}")
            if 'masks' in output:
                print(f"   masks 数量: {len(output['masks'])}")
            if 'scores' in output:
                print(f"   scores 数量: {len(output['scores'])}")
        elif isinstance(output, list):
            print(f"   列表长度: {len(output)}")

        # 找到可用的模型了
        print()
        print("=" * 60)
        print("🎉 找到可用的 SAM 2 模型!")
        print(f"   模型: {model_version}")
        print("=" * 60)

        # 更新 .env.example 文件
        print()
        print("📝 建议更新 .env 文件:")
        print(f"   SAM2_MODEL_VERSION={model_version}")

        break

    except replicate.exceptions.ReplicateError as e:
        print(f"❌ 失败: {e}")
        if "not found" in str(e).lower():
            print("   → 模型不存在")
        elif "unauthorized" in str(e).lower():
            print("   → 未授权，请检查 API Token")
        elif "rate limit" in str(e).lower():
            print("   → 达到速率限制")
        else:
            print(f"   → 其他错误: {e}")

    except Exception as e:
        print(f"❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()

    print()

print()
print("=" * 60)
print("  调试完成")
print("=" * 60)
