#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试最新模型 API 连接
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 测试配置
TEST_MODELS = [
    {
        "display": "GPT-5.2-pro",
        "model": "gpt-5.2-pro",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    },
    {
        "display": "Claude-Sonnet-4-6",
        "model": "claude-sonnet-4-6",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    },
    {
        "display": "Gemini-3.1-Pro-Preview-Thinking",
        "model": "gemini-3.1-pro-preview-thinking",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
    },
]

def test_model(model_config):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"[*] 测试: {model_config['display']}")
    print(f"    模型名称: {model_config['model']}")
    print(f"    base_url: {model_config['base_url']}")

    if not model_config['api_key']:
        print(f"    [!] 跳过: API key 未设置")
        return False

    try:
        client = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['base_url']
        )

        response = client.chat.completions.create(
            model=model_config['model'],
            messages=[{"role": "user", "content": "你好，请简短回复OK"}],
            max_tokens=10
        )

        result = response.choices[0].message.content.strip()
        print(f"    [+] 成功!")
        print(f"    响应: {result}")
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"    [!] 失败: {error_msg[:200]}")

        if "401" in error_msg or "Unauthorized" in error_msg:
            print(f"    原因: API key 无效或未设置")
        elif "404" in error_msg or "Not Found" in error_msg:
            print(f"    原因: 模型名称不存在")
        elif "400" in error_msg or "Invalid" in error_msg:
            print(f"    原因: 请求参数无效")
        return False


def main():
    print("="*60)
    print("测试最新模型 API")
    print("="*60)

    # 检查环境变量
    print("\n[*] 环境变量状态:")
    env_vars = {
        "OPENAI_API_KEY": "GPT",
        "ANTHROPIC_API_KEY": "Claude",
        "GEMINI_API_KEY": "Gemini"
    }
    for var, name in env_vars.items():
        status = "[已设置]" if os.getenv(var) else "[未设置]"
        print(f"    {var} ({name}): {status}")

    # 测试每个模型
    results = {}
    for model in TEST_MODELS:
        results[model['display']] = test_model(model)

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
        print(f"\n[!] 没有可用的模型")


if __name__ == "__main__":
    main()
