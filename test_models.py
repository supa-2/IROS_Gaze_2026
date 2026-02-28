#!/usr/bin/env python3
"""
测试中转API调用不同模型
"""
import os
import requests
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量获取API配置
BASE_URL = os.getenv("VECTOR_BASE_URL", "https://api.vectorengine.ai")
API_KEY = os.getenv("VECTOR_API_KEY", "")

# 要测试的模型
MODELS = [
    "gpt-5.2",
    "claude-sonnet-4-6",
    "gemini-3.1-pro-preview"
]

def test_model(model_name: str):
    """测试单个模型调用"""
    url = f"{BASE_URL}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Hello! Please reply with just 'OK' and the model name."}
        ],
        "max_tokens": 50
    }

    print(f"\n{'='*50}")
    print(f"Testing model: {model_name}")
    print(f"{'='*50}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Success!")
            print(f"Response: {content}")

            # 打印完整响应用于调试
            print("\nFull response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"❌ Failed!")
            print(f"Error: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"❌ Timeout after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {e}")
        return False

def main():
    """测试所有模型"""
    if not API_KEY:
        print("❌ Error: VECTOR_API_KEY not found in .env file")
        return

    print("Starting API model tests...")
    print(f"Base URL: {BASE_URL}")

    results = {}
    for model in MODELS:
        results[model] = test_model(model)

    # 总结
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {model}")

if __name__ == "__main__":
    main()
