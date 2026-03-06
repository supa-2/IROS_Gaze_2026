#!/usr/bin/env python3
"""测试 Qwen 模型可用性"""
import os
from openai import OpenAI

api_key = "sk-a90cd639de534608ab97c911022de5c1"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

models_to_test = [
    ("LLM", "qwen3.5-plus"),
    ("LLM", "qwen3.5-flash"),
    ("LLM", "qwen3.5-flash-2026-02-23"),
    ("VLM", "qwen-vl-max-latest"),
    ("VLM", "qwen-vl-plus"),
]

client = OpenAI(api_key=api_key, base_url=base_url)

for model_type, model_name in models_to_test:
    print(f"\n测试 {model_type} 模型: {model_name}")
    print("-" * 50)

    try:
        if model_type == "VLM":
            # VLM 测试（需要图片，用简单文本测试连通性）
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
                timeout=10
            )
        else:
            # LLM 测试
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
                timeout=10
            )

        result = response.choices[0].message.content
        print(f"  可用! 响应: {result}")

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid" in error_msg.lower():
            print(f"  不可用 - 认证错误或模型不存在")
        elif "timeout" in error_msg.lower():
            print(f"  不可用 - 超时")
        else:
            print(f"  不可用 - {error_msg[:100]}")

print("\n" + "=" * 50)
