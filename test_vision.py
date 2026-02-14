"""
测试图片处理和VLM集成

包括：
1. SAM 2 语义分割
2. VLM 视觉映射
"""

import sys
import os
# 设置 UTF-8 输出编码 (Windows 兼容)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from config import Config
from skills.segmentation.segmenter import SemanticSegmenter

# 测试图片路径
TEST_IMAGE = "data/test_museum.jpg"


def test_segmentation():
    """测试 SAM 2 语义分割"""
    print("=" * 60)
    print("  测试 1: SAM 2 语义分割")
    print("=" * 60)

    # 初始化配置
    config = Config()

    # 检查 API token
    if not config.model.replicate_api_token:
        print("❌ REPLICATE_API_TOKEN 未设置")
        print("   请在 .env 文件中设置 REPLICATE_API_TOKEN")
        return None

    print(f"✅ Replicate API Token: {config.model.replicate_api_token[:20]}...")
    print(f"✅ SAM 2 模型: {config.model.sam2_model_version}")

    # 检查图片文件
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ 图片文件不存在: {TEST_IMAGE}")
        return None

    print(f"✅ 测试图片: {TEST_IMAGE}")

    # 获取图片大小
    from PIL import Image
    with Image.open(TEST_IMAGE) as img:
        width, height = img.size
        print(f"   图片尺寸: {width}x{height}")

    # 初始化分割器
    print("\n⏳ 初始化 SemanticSegmenter...")
    segmenter = SemanticSegmenter(config.model)

    if not segmenter.client:
        print("❌ Replicate 客户端初始化失败")
        return None

    print("✅ Replicate 客户端已连接")

    # 执行分割
    print(f"\n⏳ 调用 SAM 2 API 分割图片...")
    print("   (这可能需要 10-30 秒)")

    try:
        result = segmenter.segment_image(TEST_IMAGE, mask_prompt="auto")

        if "error" in result:
            print(f"❌ 分割失败: {result['error']}")
            return result

        # 显示结果
        masks = result.get('masks', [])
        scores = result.get('scores', [])
        boxes = result.get('boxes', [])
        centers = result.get('centers', [])

        print(f"\n✅ 分割成功!")
        print(f"   检测到 {len(masks)} 个对象")

        for i, (box, center, score) in enumerate(zip(boxes, centers, scores)):
            print(f"   对象 {i+1}:")
            print(f"      边界框: {box}")
            print(f"      中心点: {center}")
            print(f"      置信度: {score}")

        # 导出掩码
        output_dir = "data/outputs/masks"
        print(f"\n⏳ 导出掩码到 {output_dir}...")
        saved_paths = segmenter.export_masks(result, output_dir)
        print(f"✅ 已保存 {len(saved_paths)} 个掩码文件")

        return result

    except Exception as e:
        print(f"❌ 分割过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vlm_mapping():
    """测试 VLM 视觉映射"""
    print("\n" + "=" * 60)
    print("  测试 2: VLM 视觉映射")
    print("=" * 60)

    # 检查图片文件
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ 图片文件不存在: {TEST_IMAGE}")
        return None

    print(f"✅ 测试图片: {TEST_IMAGE}")

    # 初始化 VLM
    print("\n⏳ 初始化 VLM (qwen3-omni-flash)...")
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    import base64
    import json

    config = Config()

    vlm = ChatOpenAI(
        model=config.model.vlm_model,
        api_key=config.model.llm_api_key,
        base_url=config.model.llm_base_url
    )

    print(f"✅ VLM 模型: {config.model.vlm_model}")

    # 读取并编码图片
    print("\n⏳ 读取并编码图片...")
    with open(TEST_IMAGE, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    print(f"✅ Base64 编码长度: {len(base64_image)}")

    # 构造 prompt
    prompt = """
    分析这张博物馆展厅的图片。

    请提取画面中的关键物体及其空间关系。

    重点关注：
    1. 展品 (Exhibits) - 艺术品、文物、展示品
    2. 说明牌 (Labels/Signage) - 文字说明、介绍牌
    3. 出口/通道 (Passages) - 门、通道、连接处

    请严格以 JSON 格式输出，格式如下：
    {
        "nodes": [
            {"id": "obj1", "label": "青铜鼎", "type": "Exhibit", "position": "left"},
            {"id": "obj2", "label": "说明牌", "type": "Label", "position": "center"}
        ],
        "edges": [
            {"source": "obj1", "target": "obj2", "relation": "next_to"}
        ]
    }

    只输出 JSON，不要其他内容。
    """

    # 调用 VLM
    print("\n⏳ 调用 VLM API...")
    print("   (这可能需要 10-20 秒)")

    try:
        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        response = vlm.invoke([msg])
        response_text = response.content

        print(f"\n✅ VLM 响应收到!")
        print(f"   响应长度: {len(response_text)} 字符")

        # 解析 JSON
        print("\n⏳ 解析 JSON 响应...")
        try:
            # 尝试直接解析
            result = json.loads(response_text)

            nodes = result.get('nodes', [])
            edges = result.get('edges', [])

            print(f"✅ 解析成功!")
            print(f"   检测到 {len(nodes)} 个节点:")
            for node in nodes:
                print(f"      - {node.get('label')} ({node.get('type')}) at {node.get('position')}")

            print(f"   检测到 {len(edges)} 条边:")
            for edge in edges:
                print(f"      - {edge.get('source')} -> {edge.get('target')} ({edge.get('relation')})")

            # 保存结果
            output_path = "data/outputs/vlm_result.json"
            os.makedirs("data/outputs", exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 结果已保存到: {output_path}")

            return result

        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"✅ 从响应中提取 JSON 成功!")

                nodes = result.get('nodes', [])
                edges = result.get('edges', [])

                print(f"   检测到 {len(nodes)} 个节点")
                print(f"   检测到 {len(edges)} 条边")

                output_path = "data/outputs/vlm_result.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"✅ 结果已保存到: {output_path}")

                return result
            else:
                print("⚠️  无法从响应中提取 JSON")
                print(f"\n原始响应:\n{response_text[:500]}...")
                return {"error": "Failed to parse JSON", "raw_response": response_text}

    except Exception as e:
        print(f"❌ VLM 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("  Eye-LLM 视觉模块测试")
    print("  测试图片处理 (SAM 2) 和 VLM 映射")
    print("=" * 60)

    # 创建输出目录
    os.makedirs("data/outputs/masks", exist_ok=True)
    os.makedirs("data/outputs/vlm", exist_ok=True)

    # 测试 1: SAM 2 分割
    seg_result = test_segmentation()

    # 测试 2: VLM 映射
    vlm_result = test_vlm_mapping()

    # 总结
    print("\n" + "=" * 60)
    print("  测试总结")
    print("=" * 60)

    if seg_result and 'error' not in seg_result:
        print("✅ SAM 2 语义分割: 成功")
    else:
        print("❌ SAM 2 语义分割: 失败")

    if vlm_result and 'error' not in vlm_result:
        print("✅ VLM 视觉映射: 成功")
    else:
        print("❌ VLM 视觉映射: 失败")

    print("=" * 60)


if __name__ == "__main__":
    main()
