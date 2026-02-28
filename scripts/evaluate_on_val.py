#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实验证集评估模型性能

使用 val_sharegpt.jsonl 中的 plan_scan_path 任务评估
"""

import os
import sys
import json
from typing import List, Dict
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()


class ValSetEvaluator:
    """验证集评估器"""

    def __init__(self, val_path: str = None, api_url: str = "http://localhost:8000/v1"):
        if val_path is None:
            val_path = os.path.join(
                project_root, "data", "processed", "sharegpt_json", "val_sharegpt.jsonl"
            )

        self.val_path = val_path
        self.api_url = api_url
        self.val_data = self._load_val_data()

    def _load_val_data(self) -> List[Dict]:
        """加载验证集数据"""
        data = []
        with open(self.val_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def extract_plan_scan_path_tasks(self) -> List[Dict]:
        """提取 plan_scan_path 任务"""
        tasks = []
        for item in self.val_data:
            conv = item.get('conversations', [])
            if len(conv) >= 2:
                human_msg = conv[0].get('value', '')
                gpt_msg = conv[1].get('value', '')

                # 解析任务类型
                if '"task": "plan_scan_path"' in human_msg:
                    # 解析输入
                    import re
                    json_match = re.search(r'```json\n(.+?)\n```', human_msg, re.DOTALL)
                    if json_match:
                        try:
                            request_data = json.loads(json_match.group(1))
                            exhibits = request_data.get('exhibits', [])

                            # 解析真实答案
                            json_match2 = re.search(r'```json\n(.+?)\n```', gpt_msg, re.DOTALL)
                            if json_match2:
                                try:
                                    response_data = json.loads(json_match2.group(1))
                                    scan_path = response_data.get('scan_path', [])

                                    tasks.append({
                                        'exhibits': exhibits,
                                        'ground_truth': scan_path,
                                        'raw_input': human_msg,
                                        'raw_output': gpt_msg
                                    })
                                except:
                                    continue
                        except:
                            continue

        print(f"[+] 找到 {len(tasks)} 个 plan_scan_path 任务")
        return tasks

    def extract_predict_next_tasks(self) -> List[Dict]:
        """提取 predict_next 任务"""
        tasks = []
        for item in self.val_data:
            conv = item.get('conversations', [])
            if len(conv) >= 2:
                human_msg = conv[0].get('value', '')
                gpt_msg = conv[1].get('value', '')

                if '"task": "predict_next"' in human_msg:
                    import re
                    json_match = re.search(r'```json\n(.+?)\n```', human_msg, re.DOTALL)
                    if json_match:
                        try:
                            request_data = json.loads(json_match.group(1))
                            exhibits = request_data.get('exhibits', [])
                            history = request_data.get('history', [])

                            # 解析真实答案
                            json_match2 = re.search(r'```json\n(.+?)\n```', gpt_msg, re.DOTALL)
                            if json_match2:
                                try:
                                    response_data = json.loads(json_match2.group(1))
                                    prediction = response_data.get('prediction', {})

                                    tasks.append({
                                        'exhibits': exhibits,
                                        'history': history,
                                        'ground_truth': prediction,
                                        'raw_input': human_msg,
                                        'raw_output': gpt_msg
                                    })
                                except:
                                    continue
                        except:
                            continue

        print(f"[+] 找到 {len(tasks)} 个 predict_next 任务")
        return tasks

    def evaluate_plan_scan_path(self, tasks: List[Dict], model_name: str = "Qwen") -> Dict:
        """评估 plan_scan_path 任务"""
        from openai import OpenAI
        client = OpenAI(api_key="sk-YourCustomSecretKey123", base_url=self.api_url)

        total = len(tasks)
        first_match = 0  # 第一个预测正确
        in_top_3 = 0     # 前三个预测中有正确答案
        in_top_5 = 0     # 前五个预测中有正确答案

        for i, task in enumerate(tasks):
            exhibits = task['exhibits']
            ground_truth = task['ground_truth']

            # 真实路径的展品名称列表
            gt_names = [item['name'] for item in ground_truth]

            # 构建prompt
            request_data = {
                "task": "plan_scan_path",
                "exhibits": exhibits
            }
            prompt = f"```json\n{json.dumps(request_data, ensure_ascii=False, indent=2)}\n```"

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                result = response.choices[0].message.content.strip()

                # 解析预测结果
                import re
                json_match = re.search(r'```json\n(.+?)\n```', result, re.DOTALL)
                if json_match:
                    try:
                        pred_data = json.loads(json_match.group(1))
                        pred_path = pred_data.get('scan_path', [])
                        pred_names = [item['name'] for item in pred_path]

                        # 检查第一个是否匹配
                        if pred_names and pred_names[0] == gt_names[0]:
                            first_match += 1

                        # 检查Top-3
                        if pred_names and gt_names[0] in pred_names[:3]:
                            in_top_3 += 1

                        # 检查Top-5
                        if pred_names and gt_names[0] in pred_names[:5]:
                            in_top_5 += 1

                        print(f"    [{i+1}/{total}] GT: {gt_names[0]}, Pred: {pred_names[0] if pred_names else 'None'}")

                    except:
                        print(f"    [{i+1}/{total}] Parse error")
                else:
                    print(f"    [{i+1}/{total}] No JSON found")

            except Exception as e:
                print(f"    [{i+1}/{total}] Error: {e}")

        return {
            'total': total,
            'first_match_accuracy': first_match / total if total > 0 else 0,
            'top_3_accuracy': in_top_3 / total if total > 0 else 0,
            'top_5_accuracy': in_top_5 / total if total > 0 else 0,
            'first_match_count': first_match,
            'top_3_count': in_top_3,
            'top_5_count': in_top_5
        }

    def evaluate_predict_next(self, tasks: List[Dict], model_name: str = "Qwen") -> Dict:
        """评估 predict_next 任务"""
        from openai import OpenAI
        client = OpenAI(api_key="sk-YourCustomSecretKey123", base_url=self.api_url)

        total = len(tasks)
        correct = 0

        for i, task in enumerate(tasks):
            exhibits = task['exhibits']
            history = task['history']
            ground_truth = task['ground_truth']
            gt_name = ground_truth.get('name', '')

            # 构建prompt
            request_data = {
                "task": "predict_next",
                "exhibits": exhibits,
                "history": history
            }
            prompt = f"```json\n{json.dumps(request_data, ensure_ascii=False, indent=2)}\n```"

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
                result = response.choices[0].message.content.strip()

                # 解析预测结果
                import re
                json_match = re.search(r'```json\n(.+?)\n```', result, re.DOTALL)
                if json_match:
                    try:
                        pred_data = json.loads(json_match.group(1))
                        prediction = pred_data.get('prediction', {})
                        pred_name = prediction.get('name', '')

                        if pred_name == gt_name:
                            correct += 1
                            print(f"    [{i+1}/{total}] ✓ {gt_name}")
                        else:
                            print(f"    [{i+1}/{total}] ✗ GT: {gt_name}, Pred: {pred_name}")

                    except:
                        print(f"    [{i+1}/{total}] Parse error")
                else:
                    print(f"    [{i+1}/{total}] No JSON found")

            except Exception as e:
                print(f"    [{i+1}/{total}] Error: {e}")

        return {
            'total': total,
            'accuracy': correct / total if total > 0 else 0,
            'correct_count': correct
        }

    def run_evaluation(self, model_name: str = "Qwen") -> Dict:
        """运行完整评估"""
        print("="*70)
        print("验证集评估")
        print("="*70)
        print(f"模型: {model_name}")
        print(f"验证集: {self.val_path}")
        print(f"总样本数: {len(self.val_data)}")
        print("="*70)

        results = {}

        # 评估 plan_scan_path
        print("\n[*] 评估 plan_scan_path 任务...")
        plan_tasks = self.extract_plan_scan_path_tasks()
        if plan_tasks:
            results['plan_scan_path'] = self.evaluate_plan_scan_path(plan_tasks, model_name)

        # 评估 predict_next
        print("\n[*] 评估 predict_next 任务...")
        predict_tasks = self.extract_predict_next_tasks()
        if predict_tasks:
            results['predict_next'] = self.evaluate_predict_next(predict_tasks, model_name)

        # 打印汇总
        self._print_summary(results)

        # 保存结果
        self._save_results(results, model_name)

        return results

    def _print_summary(self, results: Dict):
        """打印汇总"""
        print("\n" + "="*70)
        print("评估结果汇总")
        print("="*70)

        if 'plan_scan_path' in results:
            r = results['plan_scan_path']
            print(f"\nplan_scan_path:")
            print(f"  总样本数: {r['total']}")
            print(f"  首个匹配准确率: {r['first_match_accuracy']:.2%} ({r['first_match_count']}/{r['total']})")
            print(f"  Top-3 准确率: {r['top_3_accuracy']:.2%} ({r['top_3_count']}/{r['total']})")
            print(f"  Top-5 准确率: {r['top_5_accuracy']:.2%} ({r['top_5_count']}/{r['total']})")

        if 'predict_next' in results:
            r = results['predict_next']
            print(f"\npredict_next:")
            print(f"  总样本数: {r['total']}")
            print(f"  准确率: {r['accuracy']:.2%} ({r['correct_count']}/{r['total']})")

    def _save_results(self, results: Dict, model_name: str):
        """保存结果"""
        output_dir = "data/outputs/val_evaluation"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{model_name}_val_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[+] 结果已保存: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="验证集评估")
    parser.add_argument("--val-path", default=None, help="验证集路径")
    parser.add_argument("--api-url", default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--model", default="Qwen", help="模型名称")
    args = parser.parse_args()

    evaluator = ValSetEvaluator(args.val_path, args.api_url)
    evaluator.run_evaluation(args.model)
