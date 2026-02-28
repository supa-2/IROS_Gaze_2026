#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试最新模型 API 连接 - 使用中转API (VectorEngine)
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 统一的中转API配置
API_KEY = os.getenv("VECTOR_API_KEY", "")
BASE_URL = os.getenv("VECTOR_BASE_URL", "https://api.vectorengine.ai") + "/v1"

# 要测试的模型
TEST_MODELS = [
    "gpt-5.2-pro",
    "claude-sonnet-4-6",
    "gemini-3.1-pro-preview-thinking",
]

def test_model(model_name: str):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"[*] 测试: {model_name}")
    print(f"    base_url: {BASE_URL}")

    if not API_KEY:
        print(f"    [!] 跳过: VECTOR_API_KEY 未设置")
        return False

    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "你好，请简短回复OK和模型名称"}],
            max_tokens=50
        )

        # 打印原始响应类型
        print(f"    [DEBUG] response类型: {type(response)}")
        print(f"    [DEBUG] response内容: {response}")

        if hasattr(response, 'choices'):
            result = response.choices[0].message.content.strip()
            print(f"    [+] 成功!")
            print(f"    响应: {result}")
            return True
        else:
            print(f"    [!] 响应格式异常")
            return False

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"    [!] 异常: {error_msg[:200]}")
        print(f"    [DEBUG] traceback:\n{traceback.format_exc()}")
        return False


def main():
    print("="*60)
    print("测试最新模型 API (中转API)")
    print("="*60)
    print(f"\n[*] API配置:")
    print(f"    BASE_URL: {BASE_URL}")
    print(f"    API_KEY: {'[已设置]' if API_KEY else '[未设置]'}")

    # 测试每个模型
    results = {}
    for model in TEST_MODELS:
        results[model] = test_model(model)

    # 总结
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, success in results.items():
        status = "[✓]" if success else "[✗]"
        print(f"  {status} {name}")

    # 可用模型列表
    available = [name for name, success in results.items() if success]
    if available:
        print(f"\n[+] 可用模型: {', '.join(available)}")
    else:
        print(f"\n[!] 没有可用的模型，请检查:")
        print(f"    1. VECTOR_API_KEY 是否正确")
        print(f"    2. 模型名称是否被中转API支持")


if __name__ == "__main__":
    main()
