#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的模型推理示例 - 使用已部署的vLLM模型
"""

import os
import sys
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

# ==================== 配置 ====================
VLLM_API_URL = "http://localhost:8000/v1"  # vLLM服务地址
VLLM_API_KEY = "sk-YourCustomSecretKey123"  # vLLM默认密钥
MODEL_NAME = "Qwen"  # 模型名称

# 展品特征数据（与训练数据一致）
EXHIBIT_FEATURES = {
    "丁香花": "一幅精美的艺术画作，描绘了白色圆盆栽开满白色小花",
    "金鱼兰": "土红色盆子栽种着叶片细长、花朵呈金鱼状的植物",
    "牡丹花": "色彩饱满，花瓣层次细腻",
    "说明文字-千岛湖": "千岛湖 Qiandao Lake 1980s...",
    "玉兰花开": "开满白色玉兰花的树，挂在黑墙上",
    "人物-祝大年创作": "祝大年创作的西双版纳傣族生活主题工笔重彩人物组画",
    "自序": "白墙上陈列着的自序节选文章",
    "松竹海": "上面画着松树和竹子",
    "西双版纳": "描绘西双版纳热带雨林场景",
    "千岛湖": "湖景主题艺术作品",
    "山茶花": "一盆花的画作展品，在柱子上",
    "耕织图": "中国古代耕织图主题",
}

# 拓扑邻接关系（与训练数据一致）
TOPOLOGY_ADJACENCY = {
    "入口": ["丁香花"],
    "丁香花": ["金鱼兰", "说明文字-千岛湖"],
    "金鱼兰": ["牡丹花", "山茶花"],
    "牡丹花": ["说明文字-千岛湖"],
    "说明文字-千岛湖": ["人物-祝大年创作", "千岛湖"],
    "人物-祝大年创作": ["自序", "松竹海"],
    "自序": ["松竹海"],
    "松竹海": ["西双版纳", "耕织图"],
    "西双版纳": ["耕织图"],
    "山茶花": [],
    "千岛湖": ["耕织图"],
    "耕织图": [],
}


class GazePredictor:
    """眼动预测器 - 使用vLLM模型"""

    def __init__(self, api_url=VLLM_API_URL, api_key=VLLM_API_KEY, model_name=MODEL_NAME):
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.model_name = model_name
        print(f"[+] 已连接到vLLM服务: {api_url}")
        print(f"[+] 使用模型: {model_name}")

    def predict_next(
        self,
        current_exhibit: str,
        history: list = None,
        verbose: bool = True
    ) -> dict:
        """
        预测下一个展品

        Args:
            current_exhibit: 当前展品名称
            history: 历史记录，格式: [{"name": "展品名", "attention_level": "A"}, ...]
            verbose: 是否打印详细信息

        Returns:
            预测结果字典
        """
        if history is None:
            history = []

        # 获取候选展品（从拓扑邻接表）
        candidates = TOPOLOGY_ADJACENCY.get(current_exhibit, [])
        if not candidates:
            # 如果没有拓扑信息，使用所有展品作为候选
            candidates = list(EXHIBIT_FEATURES.keys())
            candidates = [c for c in candidates if c != current_exhibit]

        # 构建exhibits列表（ShareGPT格式，与训练数据一致）
        exhibits_list = []

        # 添加当前位置
        exhibits_list.append({
            "name": current_exhibit,
            "features": EXHIBIT_FEATURES.get(current_exhibit, "")
        })

        # 添加候选展品
        for candidate in candidates:
            exhibits_list.append({
                "name": candidate,
                "features": EXHIBIT_FEATURES.get(candidate, "")
            })

        # 构建请求数据（与训练数据格式一致）
        request_data = {
            "task": "predict_next",
            "exhibits": exhibits_list,
            "history": history
        }

        # 构建prompt
        prompt = f"```json\n{json.dumps(request_data, ensure_ascii=False, indent=2)}\n```"

        if verbose:
            print("\n" + "="*50)
            print("🔮 发送预测请求...")
            print(f"当前展品: {current_exhibit}")
            print(f"候选展品: {candidates}")
            print("="*50)

        # 调用vLLM API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300
            )

            result_text = response.choices[0].message.content.strip()

            # 解析JSON结果
            parsed = self._parse_json(result_text)

            if verbose:
                self._print_result(parsed)

            return parsed

        except Exception as e:
            print(f"[!] 预测失败: {e}")
            return {"error": str(e)}

    def predict_attention(
        self,
        exhibit_name: str,
        verbose: bool = True
    ) -> dict:
        """
        预测注意力等级

        Args:
            exhibit_name: 展品名称
            verbose: 是否打印详细信息

        Returns:
            预测结果字典
        """
        # 构建attribution任务请求数据
        request_data = {
            "task": "attribution",
            "exhibits": [
                {
                    "name": exhibit_name,
                    "features": EXHIBIT_FEATURES.get(exhibit_name, "")
                }
            ]
        }

        prompt = f"```json\n{json.dumps(request_data, ensure_ascii=False, indent=2)}\n```"

        if verbose:
            print("\n" + "="*50)
            print("👁️ 预测注意力等级...")
            print(f"展品: {exhibit_name}")
            print("="*50)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )

            result_text = response.choices[0].message.content.strip()
            parsed = self._parse_json(result_text)

            if verbose and "attribution" in parsed:
                attr = parsed["attribution"]
                print(f"\n预测结果:")
                print(f"  注意力等级: {attr.get('attention_level', 'C')}")
                print(f"  推理: {attr.get('reasoning', '')}")

            return parsed

        except Exception as e:
            print(f"[!] 预测失败: {e}")
            return {"error": str(e)}

    def _parse_json(self, text: str) -> dict:
        """从LLM输出中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass

        # 尝试提取JSON块
        depth = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx >= 0:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except:
                        continue

        return {"error": "无法解析JSON", "raw": text}

    def _print_result(self, result: dict):
        """打印预测结果"""
        print("\n📊 预测结果:")

        if "error" in result:
            print(f"  ❌ 错误: {result['error']}")
            return

        if "prediction" in result:
            pred = result["prediction"]
            print(f"  🎯 预测展品: {pred.get('name', 'N/A')}")
            print(f"  ⏱️ 预计停留: {pred.get('estimated_duration', 'N/A')}秒")
            print(f"  👁️ 注意力等级: {pred.get('attention_level', 'N/A')}")
            print(f"  💡 推理: {pred.get('reasoning', '')}")


def main():
    """主函数 - 演示如何使用"""

    # 初始化预测器
    predictor = GazePredictor()

    print("\n" + "="*60)
    print("IROS_Gaze 模型推理示例")
    print("="*60)

    # 示例1: 预测下一个展品
    print("\n【示例1】预测从'丁香花'会去哪个展品")
    result1 = predictor.predict_next(
        current_exhibit="丁香花",
        history=[
            {"name": "入口", "attention_level": "C"}
        ]
    )

    # 示例2: 带历史记录的预测
    print("\n【示例2】预测从'金鱼兰'会去哪个展品（有历史）")
    result2 = predictor.predict_next(
        current_exhibit="金鱼兰",
        history=[
            {"name": "入口", "attention_level": "C"},
            {"name": "丁香花", "attention_level": "A"}
        ]
    )

    # 示例3: 预测注意力等级
    print("\n【示例3】预测'牡丹花'的注意力等级")
    result3 = predictor.predict_attention("牡丹花")

    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)


if __name__ == "__main__":
    main()
