#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有 API 连接

测试项目中的所有外部 API 和模块连接状态
"""

import os
import sys
import json

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_config():
    """测试配置加载"""
    print_section("[1/6] 配置加载")

    try:
        from dotenv import load_dotenv
        load_dotenv()

        # 检查关键环境变量
        qwen_key = os.getenv("QWEN_API_KEY")
        qwen_base = os.getenv("QWEN_BASE_URL")
        llm_model = os.getenv("LLM_MODEL")
        vlm_model = os.getenv("VLM_MODEL")

        print(f"[*] QWEN_API_KEY: {'✓ 已配置' if qwen_key and qwen_key.startswith('sk-') else '✗ 未配置'}")
        print(f"[*] QWEN_BASE_URL: {'✓ ' + qwen_base if qwen_base else '✗ 未配置'}")
        print(f"[*] LLM_MODEL: {llm_model or '未设置'}")
        print(f"[*] VLM_MODEL: {vlm_model or '未设置'}")

        from config import Config, AttentionConfig
        config = Config()
        attention = AttentionConfig()

        print(f"[*] 注意力等级: {list(attention.ATTENTION_DURATION.keys())}")
        print(f"[+] 配置加载成功")
        return True
    except Exception as e:
        print(f"[!] 配置加载失败: {e}")
        return False


def test_topology():
    """测试拓扑引擎"""
    print_section("[2/6] 拓扑引擎")

    try:
        from skills.topology.graph_engine import TopologyEngine

        # 测试 TH 地图
        engine = TopologyEngine('TH')
        num_nodes = engine.graph.number_of_nodes()
        num_edges = engine.graph.number_of_edges()

        print(f"[*] TH 地图: {num_nodes} 个节点, {num_edges} 条边")

        # 测试查询
        result = engine.query_node('TH-E01')
        if "error" not in result:
            print(f"[*] 节点 TH-E01: {result['info']['name']}")
        else:
            print(f"[!] 节点 TH-E01 查询失败")

        print(f"[+] 拓扑引擎正常")
        return True
    except Exception as e:
        print(f"[!] 拓扑引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_system():
    """测试记忆系统"""
    print_section("[3/6] 记忆系统")

    try:
        from skills.memory.manager import MemoryManager
        from datetime import datetime

        manager = MemoryManager()

        # 添加测试记录
        manager.add_observation("TH-E01", "Test Exhibit", "A")
        manager.add_observation("TH-B02", "Another Test", "B")
        manager.add_observation("TH-E01", "Test Exhibit", "C")

        stats = manager.get_statistics()

        print(f"[*] 短期记忆: {stats['short_term_count']} 条")
        print(f"[*] 长期记忆: {stats['long_term_count']} 条")
        print(f"[*] 唯一展品: {stats['unique_exhibits']} 个")
        print(f"[*] 总时长: {stats['total_duration']} 秒")

        # 测试访问计数
        count = manager.get_visit_count("TH-E01")
        print(f"[*] TH-E01 访问次数: {count}")

        print(f"[+] 记忆系统正常")
        return True
    except Exception as e:
        print(f"[!] 记忆系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qwen_api():
    """测试 Qwen API 连接"""
    print_section("[4/6] Qwen API 连接")

    try:
        from openai import OpenAI

        api_key = os.getenv("QWEN_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            print("[!] 跳过: QWEN_API_KEY 未配置")
            return None

        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.getenv("LLM_MODEL", "qwen-plus")

        print(f"[*] API: {base_url}")
        print(f"[*] Model: {model}")

        client = OpenAI(api_key=api_key, base_url=base_url)

        # 简单测试
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "回复一个字：OK"}],
            max_tokens=10
        )

        result = response.choices[0].message.content
        print(f"[*] 响应: {result}")
        print(f"[+] Qwen API 连接成功")
        return True
    except Exception as e:
        print(f"[!] Qwen API 连接失败: {e}")
        return False


def test_vlm_api():
    """测试 VLM API 连接"""
    print_section("[5/6] VLM API 连接")

    try:
        from openai import OpenAI
        import base64

        api_key = os.getenv("QWEN_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            print("[!] 跳过: QWEN_API_KEY 未配置")
            return None

        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.getenv("VLM_MODEL", "qwen-vl-max-latest")

        print(f"[*] Model: {model}")

        # 创建一个简单的测试图片 (50x50 红色方块)
        import io
        from PIL import Image
        img = Image.new('RGB', (50, 50), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        test_image_data = base64.b64encode(buffer.read()).decode()

        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这张图片是什么颜色？只回答一个词。"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{test_image_data}"}
                        }
                    ]
                }
            ],
            max_tokens=10
        )

        result = response.choices[0].message.content
        print(f"[*] VLM 识别结果: {result}")
        print(f"[+] VLM API 连接成功")
        return True
    except Exception as e:
        print(f"[!] VLM API 连接失败: {e}")
        return False


def test_sam2_local():
    """测试 SAM2 本地模型"""
    print_section("[6/6] SAM2 本地模型")

    try:
        from skills.segmentation.sam2_local import SAM2LocalSegmenter
        import os

        # 检查模型文件
        model_path = os.path.join(project_root, "models/sam2/sam2.1_hiera_small.pt")
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"[*] 模型文件: {model_path}")
            print(f"[*] 模型大小: {size_mb:.1f} MB")

            # 测试初始化
            segmenter = SAM2LocalSegmenter(model_size='small', device='cpu')
            print(f"[+] SAM2 本地模型加载成功")
            return True
        else:
            print(f"[!] 模型文件不存在: {model_path}")
            print(f"[*] 运行以下命令下载: python scripts/download_sam2_models.py --model small")
            return False
    except Exception as e:
        print(f"[!] SAM2 本地模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("IROS Agent - API 连接测试")
    print("=" * 60)

    results = {
        'config': test_config(),
        'topology': test_topology(),
        'memory': test_memory_system(),
        'qwen_api': test_qwen_api(),
        'vlm_api': test_vlm_api(),
        'sam2_local': test_sam2_local()
    }

    # 总结
    print_section("测试总结")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for name, result in results.items():
        status = "✓ 通过" if result is True else ("✗ 失败" if result is False else "⊘ 跳过")
        print(f"  {name:20s}: {status}")

    print(f"\n总计: {passed} 通过, {failed} 失败, {skipped} 跳过")

    if failed == 0:
        print("\n[恭喜] 所有核心功能测试通过！")
    else:
        print(f"\n[注意] 有 {failed} 个测试失败，请检查配置")

    print("=" * 60)


if __name__ == "__main__":
    main()
