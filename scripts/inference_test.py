#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集推理脚本 - 生成标准格式预测结果

输出格式：
- subject_id
- episode_id
- top1, top2, top3, top4, top5 (预测的下一展品)
- p_top1 ... p_top5 (top_k对应概率)
- dwell_sec_pred (预测停留时间)
"""

import os
import sys
import json
import csv
import argparse
from typing import List, Dict, Optional
from collections import defaultdict

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


class TestSetInference:
    """测试集推理器"""

    def __init__(self, test_path: str, api_url: str = "http://localhost:8000/v1", model_name: str = "Qwen"):
        self.test_path = test_path
        self.api_url = api_url
        self.model_name = model_name

        # 初始化客户端
        from openai import OpenAI
        self.client = OpenAI(api_key="sk-YourCustomSecretKey123", base_url=api_url)

    def load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        test_data = []

        if self.test_path.endswith('.jsonl'):
            # JSONL 格式
            with open(self.test_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            # 解析出 predict_next 任务
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

                                            # 生成唯一的 episode_id
                                            subject_id = "val_subject"
                                            episode_id = f"episode_{line_no}"

                                            test_data.append({
                                                'subject_id': subject_id,
                                                'episode_id': episode_id,
                                                'exhibits': exhibits,
                                                'history': history,
                                                'line_no': line_no
                                            })
                                        except:
                                            continue
                        except:
                            continue
        elif self.test_path.endswith('.json'):
            # JSON 格式
            with open(self.test_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item_no, item in enumerate(data):
                    conv = item.get('conversations', [])
                    if len(conv) >= 2:
                        human_msg = conv[0].get('value', '')
                        if '"task": "predict_next"' in human_msg:
                            import re
                            json_match = re.search(r'```json\n(.+?)\n```', human_msg, re.DOTALL)
                            if json_match:
                                try:
                                    request_data = json.loads(json_match.group(1))
                                    exhibits = request_data.get('exhibits', [])
                                    history = request_data.get('history', [])

                                    subject_id = "test_subject"
                                    episode_id = f"episode_{item_no}"

                                    test_data.append({
                                        'subject_id': subject_id,
                                        'episode_id': episode_id,
                                        'exhibits': exhibits,
                                        'history': history,
                                        'item_no': item_no
                                    })
                                except:
                                    continue

        print(f"[+] 加载了 {len(test_data)} 个测试样本")
        return test_data

    def predict_top_k(self, exhibits: List[Dict], history: List[Dict], k: int = 5) -> List[Dict]:
        """预测 Top-K 展品及其概率"""
        # 构建 ShareGPT 格式 prompt
        request_data = {
            "task": "predict_next",
            "exhibits": exhibits,
            "history": history
        }
        prompt = f"```json\n{json.dumps(request_data, ensure_ascii=False, indent=2)}\n```"

        # 多次采样获取概率分布
        num_samples = 5
        predictions_count = defaultdict(int)

        for _ in range(num_samples):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
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

                if parsed and 'prediction' in parsed:
                    pred_name = parsed['prediction'].get('name')
                    pred_attention = parsed['prediction'].get('attention_level', 'C')
                    predictions_count[pred_name] += 1

            except Exception as e:
                continue

        # 转换为概率分布
        total = sum(predictions_count.values())
        predictions_with_prob = []
        for name, count in predictions_count.items():
            prob = count / total if total > 0 else 0
            predictions_with_prob.append({'name': name, 'probability': prob})

        # 按概率排序，取 Top-K
        predictions_with_prob.sort(key=lambda x: -x['probability'])

        # 填充不足 K 个的情况
        all_exhibit_names = [e['name'] for e in exhibits]
        for name in all_exhibit_names:
            if name not in [p['name'] for p in predictions_with_prob]:
                if len(predictions_with_prob) < k:
                    predictions_with_prob.append({'name': name, 'probability': 0.0})

        return predictions_with_prob[:k]

    def predict_attention(self, exhibits: List[Dict]) -> Dict:
        """预测注意力等级"""
        # 选择第一个非当前展品进行预测
        target_exhibit = None
        target_features = ""

        for exhibit in exhibits:
            name = exhibit.get('name', '')
            if name:  # 简化：用第一个有名字的展品
                target_exhibit = name
                target_features = exhibit.get('features', '')
                break

        if not target_exhibit:
            return {'attention_level': 'C', 'duration': 30}

        # 构建 attribution prompt
        request_data = {
            "task": "attribution",
            "exhibits": [{"name": target_exhibit, "features": target_features}]
        }
        prompt = f"```json\n{json.dumps(request_data, ensure_ascii=False, indent=2)}\n```"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
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

            if parsed and 'attribution' in parsed:
                attr = parsed['attribution']
                attention = attr.get('attention_level', 'C')
                if attention not in ATTENTION_DURATION:
                    attention = 'C'
                duration = ATTENTION_DURATION[attention]
                return {'attention_level': attention, 'duration': duration}

        except:
            pass

        return {'attention_level': 'C', 'duration': ATTENTION_DURATION['C']}

    def run_inference(self, output_path: str = None) -> List[Dict]:
        """运行推理"""
        print("="*70)
        print("测试集推理")
        print("="*70)
        print(f"测试集: {self.test_path}")
        print(f"模型: {self.model_name}")
        print(f"API: {self.api_url}")
        print("="*70)

        # 加载测试数据
        test_data = self.load_test_data()

        if not test_data:
            print("[!] 没有找到测试数据")
            return []

        # 运行推理
        predictions = []

        for i, item in enumerate(test_data):
            subject_id = item['subject_id']
            episode_id = item['episode_id']
            exhibits = item['exhibits']
            history = item.get('history', [])

            print(f"\n[{i+1}/{len(test_data)}] {subject_id} / {episode_id}")
            print(f"    展品数: {len(exhibits)}")

            # 预测 Top-5
            top_k = self.predict_top_k(exhibits, history, k=5)

            # 预测停留时间 (使用当前展品的注意力)
            current_exhibit = exhibits[0] if exhibits else {}
            attention_pred = self.predict_attention([current_exhibit])
            dwell_sec_pred = attention_pred['duration']

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
                    pred_row[f'p_top{j+1}'] = top_k[j]['probability']
                else:
                    pred_row[f'top{j+1}'] = ""
                    pred_row[f'p_top{j+1}'] = 0.0

            print(f"    Top-1: {pred_row['top1']} ({pred_row['p_top1']:.2f})")
            print(f"    停留时间: {dwell_sec_pred}s")

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
    parser.add_argument("--test-path", required=True, help="测试集路径 (json/jsonl)")
    parser.add_argument("--api-url", default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--model", default="Qwen", help="模型名称")
    parser.add_argument("--output", default=None, help="输出文件路径")
    args = parser.parse_args()

    inferencer = TestSetInference(args.test_path, args.api_url, args.model)
    inferencer.run_inference(args.output)
