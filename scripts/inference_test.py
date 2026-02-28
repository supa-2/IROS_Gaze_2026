#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集推理脚本 - 生成标准格式预测结果

参考 ablation_study.py 的调用方式
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
import numpy as np
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


class TestSetInference:
    """测试集推理器 - 使用与 ablation_study.py 相同的调用方式"""

    def __init__(self, map_name: str = "TH", api_url: str = None):
        self.map_name = map_name

        # 导入必要模块
        from skills.topology.graph_engine import TopologyEngine
        from skills.prediction.llm_reasoner import LLMReasoner
        from config import Config, AttentionConfig

        self.TopologyEngine = TopologyEngine
        self.LLMReasoner = LLMReasoner
        self.config_cls = Config
        self.AttentionConfig = AttentionConfig

        # 初始化拓扑引擎
        self.topology = TopologyEngine(map_name)
        self.all_exhibits = list(self.topology.graph.nodes())

        print(f"[+] 拓扑引擎初始化完成: {map_name}")
        print(f"[+] 展品数量: {len(self.all_exhibits)}")

    def load_test_data_from_jsonl(self, jsonl_path: str) -> List[Dict]:
        """从 JSONL 文件加载测试数据"""
        test_data = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                if line.strip():
                    try:
                        item = json.loads(line)
                        conv = item.get('conversations', [])
                        if len(conv) >= 2:
                            human_msg = conv[0].get('value', '')
                            gpt_msg = conv[1].get('value', '')

                            # 只处理 predict_next 任务
                            if '"task": "predict_next"' in human_msg:
                                import re
                                json_match = re.search(r'```json\n(.+?)\n```', human_msg, re.DOTALL)
                                if json_match:
                                    try:
                                        request_data = json.loads(json_match.group(1))
                                        exhibits = request_data.get('exhibits', [])
                                        history = request_data.get('history', [])

                                        # 从 exhibits 中获取当前展品（第一个）
                                        if exhibits:
                                            current_exhibit_name = exhibits[0].get('name', '')
                                            # 查找对应的 exhibit_id
                                            current_id = self._find_exhibit_id_by_name(current_exhibit_name)

                                            if current_id:
                                                # 获取候选展品（exhibits 除去第一个）
                                                candidates = exhibits[1:] if len(exhibits) > 1 else []

                                                subject_id = "val_subject"
                                                episode_id = f"episode_{line_no}"

                                                test_data.append({
                                                    'subject_id': subject_id,
                                                    'episode_id': episode_id,
                                                    'current_id': current_id,
                                                    'current_name': current_exhibit_name,
                                                    'candidates': candidates,
                                                    'history': history,
                                                    'line_no': line_no
                                                })
                                    except Exception as e:
                                        print(f"    [!] 解析错误 line {line_no}: {e}")
                                        continue
                    except Exception as e:
                        continue

        print(f"[+] 加载了 {len(test_data)} 个 predict_next 测试样本")
        return test_data

    def _find_exhibit_id_by_name(self, name: str) -> Optional[str]:
        """根据名称查找展品 ID"""
        # 精确匹配
        for node_id in self.all_exhibits:
            node_info = self.topology.query_node(node_id)
            if node_info and node_info.get('info', {}).get('name') == name:
                return node_id

        # 模糊匹配（包含）
        for node_id in self.all_exhibits:
            node_info = self.topology.query_node(node_id)
            if node_info:
                node_name = node_info.get('info', {}).get('name', '')
                if name in node_name or node_name in name:
                    return node_id

        return None

    def _get_candidate_ids(self, candidates: List[Dict]) -> List[str]:
        """获取候选展品 ID 列表"""
        candidate_ids = []
        for cand in candidates:
            name = cand.get('name', '')
            cand_id = self._find_exhibit_id_by_name(name)
            if cand_id:
                candidate_ids.append(cand_id)
        return candidate_ids

    def predict_top_k(self, current_id: str, candidate_ids: List[str], k: int = 5) -> List[Dict]:
        """预测 Top-K 展品及其概率"""
        if not candidate_ids:
            return []

        # 获取当前展品信息
        current_info = self.topology.query_node(current_id)

        # 构建上下文
        context = {
            'current': {
                'id': current_id,
                'name': current_info['info']['name'],
                'features': current_info['info'].get('features', ''),
                'attention_level': 'C',
                'estimated_duration': 30
            },
            'history': [],
            'spatial': {
                'previous_path': [],
                'next_path': [],
                'reachable_options': []
            }
        }

        # 添加候选展品到空间上下文
        for cand_id in candidate_ids:
            cand_info = self.topology.query_node(cand_id)
            if cand_info:
                context['spatial']['reachable_options'].append({
                    'id': cand_id,
                    'relation': 'next',
                    'name': cand_info['info']['name']
                })

        # 使用 LLMReasoner 进行预测
        attention_config = self.AttentionConfig()
        config = self.config_cls()

        reasoner = self.LLMReasoner(config.model)

        # 多次采样获取概率分布
        num_samples = 5
        predictions_count = Counter()

        for _ in range(num_samples):
            try:
                prediction = reasoner.predict_next(context, attention_config)
                pred_id = prediction.get('prediction_id')
                if pred_id and pred_id in candidate_ids:
                    predictions_count[pred_id] += 1
            except Exception as e:
                continue

        # 转换为概率分布
        total = sum(predictions_count.values())
        predictions_with_prob = []
        for cand_id in candidate_ids:
            count = predictions_count.get(cand_id, 0)
            prob = count / total if total > 0 else 0
            cand_info = self.topology.query_node(cand_id)
            name = cand_info['info']['name'] if cand_info else cand_id
            predictions_with_prob.append({'id': cand_id, 'name': name, 'probability': prob})

        # 按概率排序
        predictions_with_prob.sort(key=lambda x: -x['probability'])

        return predictions_with_prob[:k]

    def predict_dwell_time(self, exhibit_id: str, predicted_attention: str = None) -> int:
        """预测停留时间"""
        # 简化：返回默认的停留时间
        # 可以扩展为使用 attribution 任务预测
        if predicted_attention and predicted_attention in ATTENTION_DURATION:
            return ATTENTION_DURATION[predicted_attention]
        return ATTENTION_DURATION['C']  # 默认 30 秒

    def run_inference(self, test_path: str, output_path: str = None, max_samples: int = None) -> List[Dict]:
        """运行推理"""
        print("="*70)
        print("测试集推理")
        print("="*70)
        print(f"测试集: {test_path}")
        print(f"地图: {self.map_name}")
        print("="*70)

        # 加载测试数据
        test_data = self.load_test_data_from_jsonl(test_path)

        if not test_data:
            print("[!] 没有找到测试数据")
            return []

        if max_samples:
            test_data = test_data[:max_samples]
            print(f"[!] 限制样本数: {max_samples}")

        # 运行推理
        predictions = []

        for i, item in enumerate(test_data):
            subject_id = item['subject_id']
            episode_id = item['episode_id']
            current_id = item['current_id']
            candidate_ids = self._get_candidate_ids(item['candidates'])

            print(f"\n[{i+1}/{len(test_data)}] {subject_id} / {episode_id}")
            print(f"    当前: {item['current_name']} ({current_id})")
            print(f"    候选数: {len(candidate_ids)}")

            if not candidate_ids:
                print(f"    [!] 没有找到候选展品，跳过")
                continue

            # 预测 Top-5
            top_k = self.predict_top_k(current_id, candidate_ids, k=5)

            if not top_k:
                print(f"    [!] 预测失败，跳过")
                continue

            # 预测停留时间
            dwell_sec_pred = self.predict_dwell_time(current_id)
            if top_k:
                # 使用 top1 的注意力等级
                # 这里简化处理，实际可以调用 attribution

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
    parser.add_argument("--map", default="TH", help="地图名称 (TH/OS)")
    parser.add_argument("--output", default=None, help="输出文件路径")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数（用于测试）")
    args = parser.parse_args()

    inferencer = TestSetInference(map_name=args.map)
    inferencer.run_inference(args.test_path, args.output, args.max_samples)
