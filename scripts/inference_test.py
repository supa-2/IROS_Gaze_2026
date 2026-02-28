#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集推理脚本 - 直接调用模型 API

类似训练后的模型在测试集上推理，输出标准格式 predictions_test.csv
"""

import os
import sys
import json
import csv
import argparse
from typing import List, Dict, Optional
from collections import defaultdict, Counter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

# 注意力等级对应停留时间
ATTENTION_DURATION = {
    'A': 120,
    'B': 60,
    'C': 30,
    'D': 15,
    'E': 5,
}


class SimpleTestInference:
    """简单测试集推理器 - 直接调用 vLLM API"""

    def __init__(self, api_url: str = "http://localhost:8000/v1", model_name: str = "Qwen"):
        self.api_url = api_url
        self.model_name = model_name

        # 初始化 OpenAI 客户端
        from openai import OpenAI
        self.client = OpenAI(api_key="sk-YourCustomSecretKey123", base_url=api_url)

        print(f"[+] 模型: {model_name}")
        print(f"[+] API: {api_url}")

    def load_predict_next_tasks(self, jsonl_path: str, max_samples: int = None) -> List[Dict]:
        """加载 predict_next 任务"""
        tasks = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                if line.strip():
                    try:
                        item = json.loads(line)
                        conv = item.get('conversations', [])
                        if len(conv) >= 2:
                            human_msg = conv[0].get('value', '')

                            # 只处理 predict_next 任务
                            if '"task": "predict_next"' in human_msg:
                                import re
                                json_match = re.search(r'```json\n(.+?)\n```', human_msg, re.DOTALL)
                                if json_match:
                                    try:
                                        request_data = json.loads(json_match.group(1))
                                        tasks.append({
                                            'subject_id': 'val_subject',
                                            'episode_id': f'episode_{line_no}',
                                            'prompt_data': request_data,
                                            'line_no': line_no
                                        })
                                    except:
                                        continue

                        if max_samples and len(tasks) >= max_samples:
                            break
                    except:
                        continue

        print(f"[+] 加载了 {len(tasks)} 个 predict_next 任务")
        return tasks

    def predict_single(self, prompt_data: Dict) -> Dict:
        """单次预测"""
        # 构建 ShareGPT 格式的 prompt
        prompt = f"```json\n{json.dumps(prompt_data, ensure_ascii=False, indent=2)}\n```"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            result = response.choices[0].message.content.strip()

            # 解析 JSON
            depth = 0
            start_idx = -1
            parsed = None
            for i, char in enumerate(result):
                if char == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start_idx >= 0:
                        json_str = result[start_idx:i+1]
                        try:
                            parsed = json.loads(json_str)
                            break
                        except:
                            continue

            return parsed
        except Exception as e:
            return None

    def predict_top_k_with_sampling(self, prompt_data: Dict, k: int = 5) -> List[Dict]:
        """多次采样获取 Top-K 预测"""
        num_samples = 5
        predictions_count = Counter()
        attention_counts = Counter()

        # 获取候选展品列表
        exhibits = prompt_data.get('exhibits', [])
        candidate_names = [e['name'] for e in exhibits[1:]] if len(exhibits) > 1 else []

        for sample_idx in range(num_samples):
            parsed = self.predict_single(prompt_data)

            if parsed and 'prediction' in parsed:
                pred = parsed['prediction']
                pred_name = pred.get('name', '')
                pred_attention = pred.get('attention_level', 'C')

                if pred_name:
                    predictions_count[pred_name] += 1
                    attention_counts[pred_name] += 1  # 累积注意力等级

        # 转换为概率分布
        total = sum(predictions_count.values())
        predictions_with_prob = []

        for name in candidate_names:
            count = predictions_count.get(name, 0)
            prob = count / total if total > 0 else 0
            predictions_with_prob.append({'name': name, 'probability': prob})

        # 按概率排序
        predictions_with_prob.sort(key=lambda x: -x['probability'])

        # 获取预测最多的注意力等级
        most_common_attention = 'C'
        if attention_counts:
            most_common_attention = attention_counts.most_common(1)[0][0]

        return predictions_with_prob[:k], most_common_attention

    def run_inference(self, test_path: str, output_path: str = None, max_samples: int = None) -> List[Dict]:
        """运行推理"""
        print("="*70)
        print("测试集推理 - 直接调用模型")
        print("="*70)
        print(f"测试集: {test_path}")
        print("="*70)

        # 加载任务
        tasks = self.load_predict_next_tasks(test_path, max_samples)

        if not tasks:
            print("[!] 没有找到任务")
            return []

        # 运行推理
        predictions = []

        for i, task in enumerate(tasks):
            subject_id = task['subject_id']
            episode_id = task['episode_id']
            prompt_data = task['prompt_data']

            print(f"\n[{i+1}/{len(tasks)}] {episode_id}")

            # 预测 Top-5
            top_k, attention = self.predict_top_k_with_sampling(prompt_data, k=5)

            # 预测停留时间
            dwell_sec_pred = ATTENTION_DURATION.get(attention, ATTENTION_DURATION['C'])

            # 构建预测结果
            pred_row = {
                'subject_id': subject_id,
                'episode_id': episode_id,
                'dwell_sec_pred': dwell_sec_pred
            }

            # 添加 Top-5 展品和概率
            for j in range(5):
                if j < len(top_k):
                    pred_row[f'top{j+1}'] = top_k[j]['name']
                    pred_row[f'p_top{j+1}'] = round(top_k[j]['probability'], 4)
                else:
                    pred_row[f'top{j+1}'] = ""
                    pred_row[f'p_top{j+1}'] = 0.0

            print(f"    Top-1: {pred_row['top1']} ({pred_row['p_top1']:.4f})")
            print(f"    注意力: {attention}, 停留: {dwell_sec_pred}s")

            predictions.append(pred_row)

        # 保存结果
        if output_path is None:
            output_dir = "data/outputs/test_predictions"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "predictions_test.csv")

        self._save_predictions(predictions, output_path)

        return predictions

    def _save_predictions(self, predictions: List[Dict], output_path: str):
        """保存预测结果"""
        if not predictions:
            print("[!] 没有预测结果可保存")
            return

        # 字段顺序
        fieldnames = [
            'subject_id', 'episode_id',
            'top1', 'top2', 'top3', 'top4', 'top5',
            'p_top1', 'p_top2', 'p_top3', 'p_top4', 'p_top5',
            'dwell_sec_pred'
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)

        print(f"\n[+] 预测结果已保存: {output_path}")
        print(f"    总样本数: {len(predictions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试集推理")
    parser.add_argument("--test-path", required=True, help="测试集路径 (jsonl)")
    parser.add_argument("--api-url", default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--model", default="Qwen", help="模型名称")
    parser.add_argument("--output", default=None, help="输出文件路径")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数（用于测试）")
    args = parser.parse_args()

    inferencer = SimpleTestInference(args.api_url, args.model)
    inferencer.run_inference(args.test_path, args.output, args.max_samples)
