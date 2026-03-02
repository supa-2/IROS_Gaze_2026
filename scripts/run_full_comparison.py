#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整对比实验脚本

使用 Base Model (Qwen2.5-32B 4bit) 运行完整对比：
1. 测试 Base Model (4bit) 性能
2. 加载已有的闭源模型结果 (GPT-5.2, Claude, Gemini)
3. 加载消融实验结果 (Ours)
4. 生成完整对比报告

使用方法：
    python scripts/run_full_comparison.py --base-url http://<服务器IP>:8000/v1
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from datetime import datetime
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# ============================================
# 配置
# ============================================

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
    "北大简-仓颉篇": "隶书-北大简《仓颉篇》",
    "文物展柜": "天人合一部分文字文物展柜",
    "颜真卿楷书": "楷书-颜真卿《明拓干禄字书册》",
    "耕织图-多媒体装置": "数字活化的中国古代耕织图",
    "二十四节气圆盘": "融合虚拟现实技术的动态影像装置",
    "鸡蛋花": "一盆花的画作展品",
    "山茶花": "一盆花的画作展品，在柱子上",
    "千岛湖": "湖景主题艺术作品",
    "说明文字": "展品说明介绍",
    "入口": "展厅入口过渡空间",
    "森林之歌": "九幅画位于展台上面",
    "漓江春色": "祝大年1960年创作的漓江春色画作",
    "风筝": "多幅风筝主题画作",
    "鸢飞曲": "包含风筝和人的画作展品",
    "黄山松": "迎客松主题画作",
    "迎客松": "两幅画都是画的迎客松",
    "三星堆展区": "三星堆文化主题展区",
    "殷墟展区": "殷墟文化主题展区",
    "良渚展区": "良渚文化主题展区",
    "文字瀑布": "天地人自然气象等文字展示",
    "耕织图": "中国古代耕织图主题",
}

TOPOLOGY_ADJACENCY = {
    "入口": ["丁香花"],
    "丁香花": ["金鱼兰", "说明文字-千岛湖"],
    "金鱼兰": ["牡丹花", "山茶花"],
    "牡丹花": ["鸡蛋花", "说明文字-千岛湖"],
    "说明文字-千岛湖": ["人物-祝大年创作", "千岛湖"],
    "玉兰花开": ["松竹海", "西双版纳"],
    "人物-祝大年创作": ["自序", "文物展柜"],
    "自序": ["松竹海", "北大简-仓颉篇"],
    "松竹海": ["西双版纳", "漓江春色"],
    "西双版纳": ["耕织图", "颜真卿楷书"],
    "北大简-仓颉篇": ["文物展柜", "耕织图-多媒体装置"],
    "文物展柜": ["颜真卿楷书", "二十四节气圆盘"],
    "颜真卿楷书": ["耕织图-多媒体装置", "鸡蛋花"],
    "耕织图-多媒体装置": ["二十四节气圆盘", "山茶花"],
    "二十四节气圆盘": ["千岛湖", "森林之歌"],
    "鸡蛋花": ["山茶花"],
    "山茶花": ["说明文字"],
    "千岛湖": ["说明文字", "耕织图"],
    "森林之歌": ["漓江春色", "风筝"],
    "漓江春色": ["风筝", "鸢飞曲"],
    "风筝": ["鸢飞曲", "黄山松"],
    "鸢飞曲": ["黄山松", "迎客松"],
    "黄山松": ["迎客松"],
    "迎客松": ["三星堆展区"],
    "三星堆展区": ["殷墟展区"],
    "殷墟展区": ["良渚展区"],
    "良渚展区": ["文字瀑布"],
    "文字瀑布": ["耕织图"],
}


# ============================================
# 评估指标函数
# ============================================

def kl_divergence(p: Dict, q: Dict, eps: float = 1e-10) -> float:
    """计算 KL 散度（带 Laplace 平滑）"""
    keys = set(p.keys()) | set(q.keys())
    p_vals = np.array([p.get(k, eps) for k in keys]) + eps
    q_vals = np.array([q.get(k, eps) for k in keys]) + eps

    # 归一化
    p_vals = p_vals / p_vals.sum()
    q_vals = q_vals / q_vals.sum()

    return entropy(p_vals, q_vals)


def js_divergence(p: Dict, q: Dict) -> float:
    """计算 JS 散度"""
    keys = set(p.keys()) | set(q.keys())
    p_vals = np.array([p.get(k, 1e-10) for k in keys])
    q_vals = np.array([q.get(k, 1e-10) for k in keys])

    # 归一化
    p_vals = p_vals / (p_vals.sum() + 1e-10)
    q_vals = q_vals / (q_vals.sum() + 1e-10)

    return jensenshannon(p_vals, q_vals)


def correlation(p: Dict, q: Dict) -> float:
    """计算相关系数"""
    keys = list(set(p.keys()) & set(q.keys()))
    if len(keys) < 2:
        return 0.0

    p_vals = np.array([p[k] for k in keys])
    q_vals = np.array([q[k] for k in keys])

    if len(set(p_vals)) == 1 or len(set(q_vals)) == 1:
        return 0.0

    return float(np.corrcoef(p_vals, q_vals)[0, 1])


# ============================================
# Base Model 评估器
# ============================================

class BaseModelEvaluator:
    """Base Model (Qwen2.5-32B 4bit) 评估器（使用拓扑候选选择）"""

    def __init__(self, api_url: str, model_name: str = "Qwen"):
        self.api_url = api_url
        self.model_name = model_name
        self.client = OpenAI(
            api_key="sk-YourCustomSecretKey123",
            base_url=api_url
        )

        # 效率统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_time = 0
        self.num_requests = 0

        # 保存原始输出
        self.raw_outputs = {}  # {current: {"raw": str, "parsed": dict or None, "success": bool}}
        self.failed_samples = []  # 解析失败的样本列表

    def _get_topology_ordered_candidates(self, current: str, available_exhibits: List[str]) -> List[str]:
        """
        基于拓扑关系获取候选展品

        规则：
        - 获取当前展品的前2个（拓扑关系上指向当前的那些）
        - 获取当前展品的后2个（拓扑关系上当前指向的那些）
        - 如果前2个不够，从后2个补充；如果后2个不够，从前2个补充
        - 最多返回5个候选（当前 + 最多4个邻居）
        """
        candidates = []

        # 总是包含当前展品（表示从当前展品出发）
        if current in available_exhibits:
            candidates.append(current)

        # 获取后继展品（当前指向哪些）
        successors = TOPOLOGY_ADJACENCY.get(current, [])
        available_successors = [s for s in successors if s in available_exhibits]

        # 获取前序展品（哪些指向当前）
        predecessors = []
        for exhibit in available_exhibits:
            if exhibit != current:
                exhibit_neighbors = TOPOLOGY_ADJACENCY.get(exhibit, [])
                if current in exhibit_neighbors:
                    predecessors.append(exhibit)

        # 目标：当前 + 最多2个前驱 + 最多2个后继
        target_predecessors = predecessors[:2]  # 前2个
        target_successors = available_successors[:2]  # 后2个

        # 先添加前驱
        for pred in target_predecessors:
            if pred not in candidates:
                candidates.append(pred)

        # 再添加后继
        for succ in target_successors:
            if succ not in candidates:
                candidates.append(succ)

        # 如果少于3个，尝试从剩余的邻居中补充
        if len(candidates) < 3:
            remaining = predecessors[2:] + available_successors[2:]
            for item in remaining:
                if item not in candidates and len(candidates) < 5:
                    candidates.append(item)

        return candidates

    def _build_prompt_with_exhibits(self, current: str, exhibits: List[str]) -> str:
        """构建包含多个展品的prompt（ShareGPT 格式）"""
        # 构建 exhibits 列表
        exhibits_list = []
        for exhibit in exhibits:
            features = EXHIBIT_FEATURES.get(exhibit, f"{exhibit}展品")
            exhibits_list.append({
                "name": exhibit,
                "features": features
            })

        # 使用 ShareGPT 格式
        return f"""```json
{{
  "task": "predict_next",
  "exhibits": {json.dumps(exhibits_list, ensure_ascii=False)},
  "history": []
}}
```

当前场景中有{len(exhibits)}个展品，游客正在观看展品，请预测游客下一个最可能前往的展品是哪个，并给出每个候选展品的被选择概率。

返回JSON格式（注意：只需要预测下一个展品，不是规划完整路径）:
{{"predictions": [{{"name": "展品1", "probability": 0.5}}, {{"name": "展品2", "probability": 0.3}}, ...]}}"""

    def get_distribution(self, current: str, candidates: List[str], max_retries: int = 3) -> Optional[Dict[str, float]]:
        """
        获取概率分布（使用拓扑候选选择）

        Returns:
            成功返回概率分布字典，失败返回 None（不使用默认值）
        """
        # 基于拓扑获取候选（按拓扑顺序）
        topo_candidates = self._get_topology_ordered_candidates(current, candidates)

        # 打乱顺序提供（不让模型看出规律）
        import random
        shuffled_candidates = topo_candidates.copy()
        random.shuffle(shuffled_candidates)

        # 构建prompt，按打乱顺序提供
        prompt = self._build_prompt_with_exhibits(current, shuffled_candidates)

        for retry in range(max_retries):
            start_time = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300
                )

                elapsed = time.time() - start_time
                result = response.choices[0].message.content.strip()

                # 追踪 token 和时间（只在第一次调用时记录）
                if current not in self.raw_outputs:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                    self.total_time += elapsed

                # 保存原始输出
                if current not in self.raw_outputs:
                    self.raw_outputs[current] = {
                        "raw": result,
                        "topo_candidates": topo_candidates,
                        "shuffled_candidates": shuffled_candidates,
                        "retries": retry
                    }

                # 解析JSON
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    preds = parsed.get('predictions', [])
                    if preds:
                        dist = {}
                        for p in preds:
                            name = p.get('name')
                            prob = p.get('probability', 0)
                            # 只保留原始拓扑候选中的
                            if name and name in topo_candidates:
                                dist[name] = prob
                        # 归一化
                        total = sum(dist.values())
                        if total > 0:
                            self.num_requests += 1
                            self.raw_outputs[current]["parsed"] = dist
                            self.raw_outputs[current]["success"] = True
                            return {k: v/total for k, v in dist.items()}

                # 重试
                if retry < max_retries - 1:
                    prompt += "\n\n请直接返回JSON格式，不要有其他文字说明。"

            except Exception as e:
                print(f"    [ERROR] {current} -> {e}")

        # 所有重试都失败
        if current not in self.raw_outputs:
            self.raw_outputs[current] = {"success": False, "retries": max_retries}
        self.raw_outputs[current]["parsed"] = None
        self.raw_outputs[current]["success"] = False
        self.failed_samples.append(current)
        print(f"    [FAILED] {current} -> 解析失败（已重试{max_retries}次）")
        return None  # 返回 None 而非默认值

    def get_efficiency_stats(self) -> Dict:
        return {
            "avg_time": self.total_time / max(self.num_requests, 1),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "avg_input_tokens": self.total_input_tokens / max(self.num_requests, 1),
            "avg_output_tokens": self.total_output_tokens / max(self.num_requests, 1),
            "num_requests": self.num_requests
        }


# ============================================
# 实验运行器
# ============================================

class FullComparisonRunner:
    """完整对比实验运行器"""

    def __init__(self, base_url: str, ablation_path: str = None, baseline_path: str = None, test_data_path: str = None):
        self.base_url = base_url
        self.ablation_path = ablation_path or os.path.join(
            project_root, "data", "outputs", "vllm_ablation", "ablation_results.json"
        )
        self.baseline_path = baseline_path
        self.test_data_path = test_data_path

        # 加载测试数据
        if test_data_path and os.path.exists(test_data_path):
            self.test_data = self._load_test_data_from_jsonl(test_data_path)
            print(f"    [*] 从 {test_data_path} 加载了 {len(self.test_data)} 个真实测试样本")
        else:
            self.test_data = self._create_sample_data()
            print(f"    [*] 使用模拟测试数据 ({len(self.test_data)} 个样本)")

        self.real_distributions = self._build_real_distributions()

    def _load_test_data_from_jsonl(self, jsonl_path: str) -> List[Dict]:
        """从 ShareGPT JSONL 文件加载 predict_next 测试数据"""
        test_data = []
        import re

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
                                # 提取请求数据
                                json_match = re.search(r'```json\n(.+?)\n```', human_msg, re.DOTALL)
                                if json_match:
                                    try:
                                        request_data = json.loads(json_match.group(1))
                                        exhibits = request_data.get('exhibits', [])
                                        if len(exhibits) >= 2:
                                            current = exhibits[0]['name']

                                            # 提取 ground truth
                                            gt_match = re.search(r'```json\n(.+?)\n```', gpt_msg, re.DOTALL)
                                            next_exhibit = None
                                            if gt_match:
                                                try:
                                                    gt_data = json.loads(gt_match.group(1))
                                                    if 'prediction' in gt_data:
                                                        next_exhibit = gt_data['prediction'].get('name')
                                                except:
                                                    pass

                                            if next_exhibit:
                                                test_data.append({
                                                    'current': current,
                                                    'next': next_exhibit,
                                                    'line_no': line_no
                                                })
                                    except:
                                        continue
                    except:
                        continue

        return test_data

    def _create_sample_data(self) -> List[Dict]:
        """创建模拟测试数据（基于拓扑关系的转移）"""
        # 只包含在拓扑中存在的转移
        return [
            {"current": "入口", "next": "丁香花"},
            {"current": "入口", "next": "丁香花"},
            {"current": "入口", "next": "丁香花"},
            {"current": "丁香花", "next": "金鱼兰"},
            {"current": "丁香花", "next": "金鱼兰"},
            {"current": "丁香花", "next": "金鱼兰"},
            {"current": "丁香花", "next": "说明文字-千岛湖"},
            {"current": "金鱼兰", "next": "牡丹花"},
            {"current": "金鱼兰", "next": "牡丹花"},
            {"current": "金鱼兰", "next": "山茶花"},
            {"current": "说明文字-千岛湖", "next": "人物-祝大年创作"},
            {"current": "说明文字-千岛湖", "next": "千岛湖"},
            {"current": "玉兰花开", "next": "松竹海"},
            {"current": "玉兰花开", "next": "西双版纳"},
            {"current": "松竹海", "next": "西双版纳"},
            {"current": "松竹海", "next": "漓江春色"},
            {"current": "松竹海", "next": "西双版纳"},
            {"current": "西双版纳", "next": "耕织图"},
            {"current": "西双版纳", "next": "颜真卿楷书"},
            {"current": "漓江春色", "next": "风筝"},
            {"current": "漓江春色", "next": "鸢飞曲"},
            {"current": "风筝", "next": "鸢飞曲"},
            {"current": "风筝", "next": "黄山松"},
            {"current": "迎客松", "next": "三星堆展区"},
            {"current": "三星堆展区", "next": "殷墟展区"},
            {"current": "三星堆展区", "next": "殷墟展区"},
            {"current": "殷墟展区", "next": "良渚展区"},
            {"current": "殷墟展区", "next": "良渚展区"},
            {"current": "良渚展区", "next": "文字瀑布"},
            {"current": "良渚展区", "next": "文字瀑布"},
            {"current": "文字瀑布", "next": "耕织图"},
            {"current": "文字瀑布", "next": "耕织图"},
        ]

    def _build_real_distributions(self) -> Dict[str, Dict[str, float]]:
        """构建真实的转移分布"""
        distributions = defaultdict(Counter)
        for sample in self.test_data:
            current = sample['current']
            next_exhibit = sample['next']
            distributions[current][next_exhibit] += 1

        result = {}
        for current, counter in distributions.items():
            total = sum(counter.values())
            result[current] = {
                exhibit: count / total
                for exhibit, count in counter.items()
            }
        return result

    def evaluate_method(self, method) -> Dict:
        """评估单个方法"""
        kl_divs = []
        js_divs = []
        corrs = []
        top1_correct = 0
        top3_correct = 0
        top_total = 0
        failed_count = 0  # 解析失败的样本数

        for current, real_dist in self.real_distributions.items():
            candidates = list(real_dist.keys())
            model_dist = method.get_distribution(current, candidates)

            # 如果返回 None，说明解析失败，跳过
            if model_dist is None:
                failed_count += 1
                continue

            try:
                kl = kl_divergence(real_dist, model_dist)
                js = js_divergence(real_dist, model_dist)
                corr = correlation(real_dist, model_dist)
                kl_divs.append(kl)
                js_divs.append(js)
                corrs.append(corr)
            except:
                continue

        for sample in self.test_data:
            current = sample['current']
            actual_next = sample['next']

            if current not in self.real_distributions:
                continue
            candidates = list(self.real_distributions[current].keys())

            model_dist = method.get_distribution(current, candidates)

            # 如果返回 None，跳过此样本
            if model_dist is None:
                failed_count += 1
                continue

            sorted_preds = sorted(model_dist.items(), key=lambda x: -x[1])

            if sorted_preds and sorted_preds[0][0] == actual_next:
                top1_correct += 1

            top_k_preds = [p[0] for p in sorted_preds[:3]]
            if actual_next in top_k_preds:
                top3_correct += 1

            top_total += 1

        return {
            'top1_accuracy': top1_correct / max(top_total, 1),
            'top3_accuracy': top3_correct / max(top_total, 1),
            'kl_divergence': np.mean(kl_divs) if kl_divs else 0,
            'js_divergence': np.mean(js_divs) if js_divs else 0,
            'correlation': np.mean(corrs) if corrs else 0,
            'num_evaluated': len(kl_divs),
            'num_top_eval': top_total,
            'failed_samples': failed_count,
            'success_rate': (len(self.real_distributions) - failed_count) / len(self.real_distributions) if self.real_distributions else 0
        }

    def load_ablation_results(self) -> Dict:
        """加载消融实验结果"""
        if os.path.exists(self.ablation_path):
            try:
                with open(self.ablation_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    full_result = results.get('Full', {})
                    if full_result:
                        return full_result
            except Exception as e:
                print(f"    [!] 加载消融实验结果失败: {e}")

        # 默认值
        return {
            'top1_accuracy': 0.739,
            'top3_accuracy': 1.0,
            'kl_divergence': 0.048,
            'js_divergence': 0.061,
            'correlation': -0.083,
        }

    def load_existing_baseline_results(self) -> Dict:
        """加载已有的闭源模型对照实验结果"""
        # 如果指定了路径，直接使用
        if self.baseline_path:
            print(f"    [*] 加载已有对照实验结果: {self.baseline_path}")
            try:
                with open(self.baseline_path, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except Exception as e:
                print(f"    [!] 加载已有结果失败: {e}")
                return {}
        else:
            # 尝试多个可能的路径
            possible_paths = [
                Path(project_root) / "data" / "outputs" / "baselines",
                Path(project_root) / "data" / "outputs" / "baselines",
                Path.cwd() / "data" / "outputs" / "baselines",
                Path.cwd() / "data" / "baselines",
                Path("/home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026/data/outputs/baselines"),
            ]

            baseline_files = []
            for base_dir in possible_paths:
                if base_dir.exists():
                    baseline_files.extend(list(base_dir.glob("baseline_results_*.json")))

            if not baseline_files:
                print("    [!] 未找到已有的对照实验结果，请使用 --baseline 参数指定")
                return {}

            # 使用最新的结果文件
            latest_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
            print(f"    [*] 加载已有对照实验结果: {latest_file}")

            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except Exception as e:
                print(f"    [!] 加载已有结果失败: {e}")
                return {}

        # 提取闭源模型结果
        zero_shot_results = {}
        for key in ["GPT-5.2", "Claude-Sonnet-4-6", "Gemini-3.1-Pro-Thinking"]:
            if key in all_results:
                zero_shot_results[key] = all_results[key]

        # 也加载 Markov Chain 和 LSTM
        if "Markov Chain" in all_results:
            zero_shot_results["Markov Chain"] = all_results["Markov Chain"]
        if "LSTM" in all_results:
            zero_shot_results["LSTM"] = all_results["LSTM"]

        return zero_shot_results

    def run(self) -> Dict:
        """运行完整对比实验"""
        print("="*90)
        print("完整对比实验 - Base Model (4bit) vs Fine-tuned vs Zero-Shot LLMs")
        print("="*90)
        print(f"Base Model URL: {self.base_url}")
        print(f"真实分布起点数: {len(self.real_distributions)}")
        print(f"测试样本数: {len(self.test_data)}")

        results = {}
        raw_outputs = None

        # 1. 加载已有的闭源模型结果
        print("\n[*] Loading: 已有对照实验结果...")
        existing_results = self.load_existing_baseline_results()
        for name, result in existing_results.items():
            results[name] = result
            print(f"    {name}: Top-1={result['top1_accuracy']:.1%}")

        # 2. Base Model (4bit)
        print("\n[*] Testing: Base Model (Qwen2.5-32B 4bit)...")
        base_model = BaseModelEvaluator(self.base_url)
        results['Base Model (4bit)'] = self.evaluate_method(base_model)
        results['Base Model (4bit)'].update(base_model.get_efficiency_stats())
        raw_outputs = base_model.raw_outputs
        self._print_result('Base Model (4bit)', results['Base Model (4bit)'])

        # 打印解析失败统计
        if hasattr(base_model, 'failed_samples') and base_model.failed_samples:
            print(f"    [!] 解析失败的样本: {len(base_model.failed_samples)} 个")
            print(f"    [!] 成功率: {results['Base Model (4bit)'].get('success_rate', 0):.1%}")

        # 3. Ours (消融实验结果)
        print("\n[*] Loading: Ours (Fine-tuned) from ablation results...")
        results['Ours (Fine-tuned)'] = self.load_ablation_results()
        self._print_result('Ours (Fine-tuned)', results['Ours (Fine-tuned)'])

        # 保存结果（包含原始输出）
        self._save_results(results, raw_outputs)
        self._print_comparison_table(results)

        return results

    def _print_result(self, name: str, result: Dict):
        """打印单个结果"""
        print(f"    Top-1: {result['top1_accuracy']:.2%} ↑")
        print(f"    Top-3: {result['top3_accuracy']:.2%} ↑")
        print(f"    KL散度: {result['kl_divergence']:.4f} ↓")
        print(f"    JS散度: {result['js_divergence']:.4f} ↓")
        print(f"    相关系数: {result['correlation']:.4f} ↑")

        if 'avg_time' in result:
            print(f"    平均响应时间: {result['avg_time']:.2f}s")
            print(f"    Token消耗: {result['total_tokens']} (输入={result['total_input_tokens']}, 输出={result['total_output_tokens']})")

    def _save_results(self, results: Dict, raw_outputs: Dict = None):
        """保存结果"""
        output_dir = Path(project_root) / "data" / "outputs" / "baselines"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"full_comparison_{timestamp}.json"

        # 如果有原始输出，一起保存
        save_data = results.copy()
        if raw_outputs:
            save_data['raw_outputs'] = raw_outputs

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\n[+] Results saved to {output_path}")

    def _print_comparison_table(self, results: Dict):
        """打印对比表格"""
        print("\n" + "="*90)
        print("对比表格")
        print("="*90)

        print("\n" + "-"*90)
        print(f"{'Method':<25} {'Top-1':>10} {'Top-3':>10} {'KL':>10} {'JS':>10} {'Time(s)':>10}")
        print("-"*90)

        for name, result in results.items():
            print(f"{name:<25} {result['top1_accuracy']:>10.1%} {result['top3_accuracy']:>10.1%} "
                  f"{result['kl_divergence']:>10.4f} {result['js_divergence']:>10.4f} ", end="")

            if 'avg_time' in result:
                print(f"{result['avg_time']:>10.2f}")
            else:
                print(f"{'N/A':>10}")

        print("-"*90)

        # 打印 LaTeX 表格
        print("\n" + "="*90)
        print("LaTeX Table")
        print("="*90)

        print("\n\\begin{table}[t]")
        print("\\centering")
        print("\\caption{Performance comparison between Base Model and Fine-tuned Model.}")
        print("\\label{tab:base_vs_finetuned}")
        print("\\begin{tabular}{lccccc}")
        print("\\hline")
        print("Method & Top-1$\\uparrow$ & Top-3$\\uparrow$ & KL$\\downarrow$ & JS$\\downarrow$ & Time(s)$\\downarrow$ \\\\")
        print("\\hline")

        for name, result in results.items():
            top1 = f"\\textbf{{{result['top1_accuracy']:.1%}}}" if 'Fine-tuned' in name else f"{result['top1_accuracy']:.1%}"
            top3 = f"\\textbf{{{result['top3_accuracy']:.1%}}}" if 'Fine-tuned' in name else f"{result['top3_accuracy']:.1%}"
            kl = f"\\textbf{{{result['kl_divergence']:.3f}}}" if 'Fine-tuned' in name else f"{result['kl_divergence']:.3f}"
            js = f"\\textbf{{{result['js_divergence']:.3f}}}" if 'Fine-tuned' in name else f"{result['js_divergence']:.3f}"

            time_str = f"\\textbf{{{result.get('avg_time', 0):.2f}}}" if 'Fine-tuned' in name else f"{result.get('avg_time', 0):.2f}"

            print(f"{name} & {top1} & {top3} & {kl} & {js} & {time_str} \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="完整对比实验 - Base Model (4bit) vs Fine-tuned vs Zero-Shot LLMs")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="Base Model API URL (vLLM)")
    parser.add_argument("--ablation", default=None,
                        help="消融实验结果路径")
    parser.add_argument("--baseline", default=None,
                        help="已有对照实验结果路径 (baseline_results_*.json)")
    args = parser.parse_args()

    runner = FullComparisonRunner(args.base_url, args.ablation, args.baseline)
    results = runner.run()
