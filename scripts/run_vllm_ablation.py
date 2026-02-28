#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study with Distribution Matching and Attention Evaluation

使用分布匹配度评估，而非单一正确答案
同时评估注意力等级预测能力
"""

import os
import sys
import json
import random
import numpy as np
from typing import List, Dict, Optional
from collections import Counter, defaultdict
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from datetime import datetime

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# 注意力等级配置
ATTENTION_DURATION = {
    'A': 120,  # 深度关注 - 长时间仔细观看
    'B': 60,   # 中等关注 - 正常观看
    'C': 30,   # 一般关注 - 浏览式观看
    'D': 15,   # 快速浏览 - 短暂停留
    'E': 5,    # 一瞥而过 - 快速扫视
}

ATTENTION_DESCRIPTION = {
    'A': '深度关注 - 长时间仔细观看',
    'B': '中等关注 - 正常观看',
    'C': '一般关注 - 浏览式观看',
    'D': '快速浏览 - 短暂停留',
    'E': '一瞥而过 - 快速扫视'
}

# 真实展品名称
REAL_EXHIBIT_NAMES = [
    "丁香花", "金鱼兰", "牡丹花", "说明文字-千岛湖", "玉兰花开",
    "人物-祝大年创作", "自序", "松竹海", "西双版纳",
    "北大简-仓颉篇", "文物展柜", "颜真卿楷书", "耕织图-多媒体装置",
    "二十四节气圆盘", "鸡蛋花", "山茶花", "千岛湖", "说明文字", "入口",
    "森林之歌", "漓江春色", "风筝", "鸢飞曲", "黄山松", "迎客松",
    "三星堆展区", "殷墟展区", "良渚展区", "文字瀑布", "耕织图"
]

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
}


class AblationConfig:
    """
    消融实验配置

    消融方式：
    - No-Topology: 打乱 exhibits 顺序，消除空间位置暗示
    - No-Feature: 将 features 字段设为空字符串
    - No-Memory: 将 history 设为空数组
    """
    def __init__(
        self,
        use_topology_preprocess: bool = True,
        use_feature_preprocess: bool = True,
        use_memory_in_llm: bool = True,
    ):
        self.use_topology_preprocess = use_topology_preprocess
        self.use_feature_preprocess = use_feature_preprocess
        self.use_memory_in_llm = use_memory_in_llm

    def get_name(self) -> str:
        parts = []
        if not self.use_topology_preprocess:
            parts.append("No-Topology")
        if not self.use_feature_preprocess:
            parts.append("No-Feature")
        if not self.use_memory_in_llm:
            parts.append("No-Memory")
        return "-".join(parts) if parts else "Full"


class DistributionAblation:
    """使用分布匹配度的消融实验"""

    def __init__(
        self,
        model_name: str = "Qwen",
        api_url: str = "http://localhost:8000/v1",
        api_key: str = "sk-YourCustomSecretKey123",
        data_path: str = None
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.data_path = data_path

        # 加载数据并构建真实分布
        self.test_data = self._load_test_data()
        self.real_distributions = self._build_real_distributions()

        # 初始化客户端
        self._init_client()
        self.preprocess_cache = {}
        self.num_samples = 5  # 每个预测采样5次来构建分布
        self.laplace_alpha = 1.0  # Laplace平滑系数，避免零概率导致KL散度为Infinity

    def _init_client(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        print(f"[+] Connected to vLLM API: {self.api_url}")

    def _load_test_data(self) -> List[Dict]:
        if self.data_path and os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                return json.load(f)
        return self._create_sample_data()

    def _create_sample_data(self) -> List[Dict]:
        """创建测试数据（包含注意力等级）"""
        sample_sequences = [
            ["入口", "丁香花", "金鱼兰", "牡丹花"],
            ["入口", "丁香花", "金鱼兰", "山茶花"],
            ["入口", "丁香花", "说明文字-千岛湖", "人物-祝大年创作"],
            ["入口", "说明文字-千岛湖", "千岛湖"],
            ["入口", "金鱼兰", "牡丹花"],
            ["玉兰花开", "松竹海", "西双版纳", "耕织图"],
            ["松竹海", "漓江春色", "风筝"],
            ["迎客松", "三星堆展区", "殷墟展区"],
            ["三星堆展区", "良渚展区"],
            ["良渚展区", "文字瀑布", "耕织图"],
        ]

        # 根据展品类型分配注意力等级
        exhibit_attention = {
            "入口": "C", "丁香花": "A", "金鱼兰": "B", "牡丹花": "A",
            "说明文字-千岛湖": "B", "人物-祝大年创作": "A", "千岛湖": "A",
            "山茶花": "C", "松竹海": "B", "西双版纳": "A", "耕织图": "B",
            "漓江春色": "A", "风筝": "B", "迎客松": "A", "三星堆展区": "B",
            "殷墟展区": "B", "良渚展区": "B", "文字瀑布": "C",
        }

        test_data = []
        for seq in sample_sequences:
            for i in range(len(seq) - 1):
                next_exhibit = seq[i + 1]
                attention = exhibit_attention.get(next_exhibit, "C")
                test_data.append({
                    'current': seq[i],
                    'next': next_exhibit,
                    'attention': attention,
                    'duration': ATTENTION_DURATION[attention]
                })
        return test_data

    def _build_real_distributions(self) -> Dict[str, Dict[str, float]]:
        """构建真实的转移分布"""
        distributions = defaultdict(Counter)

        for sample in self.test_data:
            current = sample['current']
            next_exhibit = sample['next']
            distributions[current][next_exhibit] += 1

        # 转换为概率分布
        result = {}
        for current, counter in distributions.items():
            total = sum(counter.values())
            result[current] = {
                exhibit: count / total
                for exhibit, count in counter.items()
            }
        return result

    def get_model_distribution(
        self,
        current: str,
        candidates: List[str],
        config: AblationConfig
    ) -> Dict[str, float]:
        """获取模型预测的分布 - 使用与训练数据一致的 ShareGPT 格式"""
        # 构建 exhibits 列表（第一个是当前位置，其余是候选）
        # 空间顺序暗示拓扑关系
        exhibits_list = []

        # 添加当前位置
        current_features = EXHIBIT_FEATURES.get(current, "")
        exhibits_list.append({
            "name": current,
            "features": current_features if config.use_feature_preprocess else ""
        })

        # 添加候选展品
        candidates_for_order = list(candidates)

        # 如果不使用拓扑，打乱候选顺序
        if not config.use_topology_preprocess:
            random.seed(42)  # 固定种子保证可复现
            random.shuffle(candidates_for_order)

        for candidate in candidates_for_order:
            feat = EXHIBIT_FEATURES.get(candidate, "")
            exhibits_list.append({
                "name": candidate,
                "features": feat if config.use_feature_preprocess else ""
            })

        # 构建历史记录（如果使用记忆）
        history = []
        if config.use_memory_in_llm:
            # 模拟历史：假设之前访问过入口
            history.append({
                "name": "入口",
                "attention_level": "C"
            })

        # 构建 ShareGPT 格式的 prompt（与训练数据一致）
        request_data = {
            "task": "predict_next",
            "exhibits": exhibits_list,
            "history": history
        }

        # 转换为 JSON 格式
        prompt = f"```json\n{json.dumps(request_data, ensure_ascii=False, indent=2)}\n```"

        # 多次采样来构建分布
        predictions_count = {c: 0 for c in candidates}

        for _ in range(self.num_samples):
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

                # 提取预测结果
                if parsed and 'prediction' in parsed:
                    pred_name = parsed['prediction'].get('name')
                    if pred_name and pred_name in candidates:
                        predictions_count[pred_name] += 1

            except Exception as e:
                continue

        # 转换为概率分布（使用 Laplace 平滑避免零概率）
        total = sum(predictions_count.values())
        if total > 0:
            # 平滑后的概率: (count + alpha) / (total + alpha * num_classes)
            smoothed = {
                c: (predictions_count[c] + self.laplace_alpha) / (total + self.laplace_alpha * len(candidates))
                for c in candidates
            }
            return smoothed

        # 默认：均匀分布（也应用平滑）
        return {c: (1 + self.laplace_alpha) / (len(candidates) + self.laplace_alpha * len(candidates)) for c in candidates}

    def predict_attention(
        self,
        next_exhibit: str,
        config: AblationConfig
    ) -> Dict[str, any]:
        """预测注意力等级和停留时间 - 使用与训练数据一致的 attribution 格式"""
        features = EXHIBIT_FEATURES.get(next_exhibit, "")

        # 选择一个对比展品（用于 attribution 任务）
        # 选择第一个不同于 next_exhibit 的展品
        compare_exhibit = None
        for name in EXHIBIT_FEATURES.keys():
            if name != next_exhibit:
                compare_exhibit = name
                break

        # 构建 attribution 任务格式（与训练数据一致）
        # 训练数据中 attribution 是比较两个展品，选择更有吸引力的
        request_data = {
            "task": "attribution",
            "exhibits": [
                {
                    "name": next_exhibit,
                    "features": features if config.use_feature_preprocess else ""
                }
            ]
        }

        # 如果有对比展品，添加它（更接近训练格式）
        if compare_exhibit:
            request_data["exhibits"].append({
                "name": compare_exhibit,
                "features": EXHIBIT_FEATURES.get(compare_exhibit, "") if config.use_feature_preprocess else ""
            })

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
                # 确保attention是有效的等级
                if attention not in ATTENTION_DURATION:
                    attention = 'C'
                duration = ATTENTION_DURATION.get(attention, ATTENTION_DURATION['C'])
                return {
                    'attention_level': attention,
                    'estimated_duration': duration,
                    'reasoning': attr.get('reasoning', '')
                }

        except Exception as e:
            print(f"[!] Attention prediction error: {e}")

        # 默认：中等关注
        return {
            'attention_level': 'C',
            'estimated_duration': ATTENTION_DURATION['C'],
            'reasoning': '默认预测'
        }

    def kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """计算 KL 散度"""
        # 确保两个分布有相同的键
        all_keys = set(p.keys()) | set(q.keys())
        eps = 1e-10

        p_vec = np.array([p.get(k, eps) for k in all_keys])
        q_vec = np.array([q.get(k, eps) for k in all_keys])

        return entropy(p_vec, q_vec)

    def js_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """计算 JS 散度"""
        all_keys = set(p.keys()) | set(q.keys())
        eps = 1e-10

        p_vec = np.array([p.get(k, eps) for k in all_keys])
        q_vec = np.array([q.get(k, eps) for k in all_keys])

        return jensenshannon(p_vec, q_vec)

    def correlation(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """计算相关系数"""
        all_keys = sorted(set(p.keys()) | set(q.keys()))
        p_vec = np.array([p.get(k, 0) for k in all_keys])
        q_vec = np.array([q.get(k, 0) for k in all_keys])

        if np.std(p_vec) == 0 or np.std(q_vec) == 0:
            return 0.0

        return np.corrcoef(p_vec, q_vec)[0, 1]

    def evaluate_config(self, config: AblationConfig) -> Dict:
        """评估单个配置"""
        print(f"    评估中...", end='', flush=True)

        kl_divs = []
        js_divs = []
        corrs = []

        # Top-1/Top-3 准确率
        top1_correct = 0
        top3_correct = 0
        top_total = 0

        # 注意力评估
        attention_correct = 0
        attention_total = 0
        duration_errors = []

        # 对每个有真实分布的起点进行评估
        for current, real_dist in self.real_distributions.items():
            candidates = list(real_dist.keys())

            # 获取模型预测的分布
            model_dist = self.get_model_distribution(current, candidates, config)

            # 计算分布指标
            try:
                kl = self.kl_divergence(real_dist, model_dist)
                js = self.js_divergence(real_dist, model_dist)
                corr = self.correlation(real_dist, model_dist)

                kl_divs.append(kl)
                js_divs.append(js)
                corrs.append(corr)
            except:
                continue

        # 对每个测试样本评估 Top-1/Top-3
        for sample in self.test_data:
            current = sample['current']
            actual_next = sample['next']

            # 获取当前展品的所有候选（从真实分布中获取）
            if current not in self.real_distributions:
                continue
            candidates = list(self.real_distributions[current].keys())

            # 获取模型预测的分布
            model_dist = self.get_model_distribution(current, candidates, config)

            # 按概率排序
            sorted_preds = sorted(model_dist.items(), key=lambda x: -x[1])

            # Top-1: 最高概率的是否匹配
            if sorted_preds and sorted_preds[0][0] == actual_next:
                top1_correct += 1

            # Top-3: 前3最高概率中是否匹配
            top_k_preds = [p[0] for p in sorted_preds[:3]]
            if actual_next in top_k_preds:
                top3_correct += 1

            top_total += 1

        # 评估注意力预测
        for sample in self.test_data:
            next_exhibit = sample['next']
            true_attention = sample.get('attention', 'C')
            true_duration = sample.get('duration', ATTENTION_DURATION['C'])

            # 预测注意力
            pred = self.predict_attention(next_exhibit, config)
            pred_attention = pred['attention_level']
            pred_duration = pred['estimated_duration']

            # 注意力等级准确率
            if pred_attention == true_attention:
                attention_correct += 1
            attention_total += 1

            # 停留时间误差
            duration_errors.append(abs(pred_duration - true_duration))

        print(f" 完成 ({len(kl_divs)} 个起点, {top_total} 个预测, {attention_total} 个注意力预测)")

        return {
            'top1_accuracy': top1_correct / max(top_total, 1),
            'top3_accuracy': top3_correct / max(top_total, 1),
            'kl_divergence': np.mean(kl_divs) if kl_divs else 0,
            'js_divergence': np.mean(js_divs) if js_divs else 0,
            'correlation': np.mean(corrs) if corrs else 0,
            'attention_accuracy': attention_correct / max(attention_total, 1),
            'duration_mae': np.mean(duration_errors) if duration_errors else 0,
            'num_evaluated': len(kl_divs),
            'num_top_eval': top_total,
            'num_attention_eval': attention_total
        }

    def run_ablation_study(self) -> Dict:
        """运行消融实验"""
        print("="*70)
        print("分布匹配度消融实验")
        print("="*70)
        print(f"模型: {self.model_name}")
        print(f"真实分布起点数: {len(self.real_distributions)}")
        print(f"测试样本数: {len(self.test_data)}")
        print("="*70)

        # 显示真实分布
        print("\n[*] 真实转移分布:")
        for current, dist in self.real_distributions.items():
            items = sorted(dist.items(), key=lambda x: -x[1])
            print(f"  {current} → {', '.join([f'{k}({v:.0%})' for k, v in items[:3]])}")

        results = {}

        configs = [
            AblationConfig(use_topology_preprocess=True, use_feature_preprocess=True, use_memory_in_llm=True),
            AblationConfig(use_topology_preprocess=False, use_feature_preprocess=True, use_memory_in_llm=True),
            AblationConfig(use_topology_preprocess=True, use_feature_preprocess=False, use_memory_in_llm=True),
            AblationConfig(use_topology_preprocess=True, use_feature_preprocess=True, use_memory_in_llm=False),
        ]

        for config in configs:
            config_name = config.get_name()
            print(f"\n[*] 测试: {config_name}")

            config_results = self.evaluate_config(config)
            results[config_name] = config_results

            print(f"    Top-1: {config_results['top1_accuracy']:.2%} ↑ (越高越好)")
            print(f"    Top-3: {config_results['top3_accuracy']:.2%} ↑ (越高越好)")
            print(f"    KL散度: {config_results['kl_divergence']:.4f} ↓ (越低越好)")
            print(f"    JS散度: {config_results['js_divergence']:.4f} ↓ (越低越好)")
            print(f"    相关系数: {config_results['correlation']:.4f} ↑ (越高越好)")
            print(f"    注意力准确率: {config_results['attention_accuracy']:.2%} ↑ (越高越好)")
            print(f"    停留时间MAE: {config_results['duration_mae']:.1f}s ↓ (越低越好)")

        self._save_results(results)
        self._print_table(results)
        self._print_latex_table(results)
        return results

    def _save_results(self, results: Dict):
        """保存结果"""
        output_dir = "data/outputs/vllm_ablation"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ablation_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[+] 结果已保存: {output_path}")

    def _print_table(self, results: Dict):
        """打印结果表格"""
        print("\n" + "="*90)
        print("消融实验结果 - Top准确率 + 分布匹配度 + 注意力预测")
        print("="*90)
        print(f"{'配置':<20} {'Top-1↑':>10} {'Top-3↑':>10} {'KL↓':>10} {'JS↓':>10} {'Corr↑':>10} {'Attn↑':>10} {'MAE↓':>10}")
        print("-" * 90)

        full = results.get('Full', {})
        print(f"{'Full (Ours)':<20} {full['top1_accuracy']:>10.2%} {full['top3_accuracy']:>10.2%} "
              f"{full['kl_divergence']:>10.4f} {full['js_divergence']:>10.4f} "
              f"{full['correlation']:>10.4f} {full['attention_accuracy']:>10.2%} {full['duration_mae']:>10.1f}s")

        for name, res in results.items():
            if name == 'Full':
                continue
            diff_top1 = full['top1_accuracy'] - res['top1_accuracy']
            diff_top3 = full['top3_accuracy'] - res['top3_accuracy']
            diff_kl = res['kl_divergence'] - full['kl_divergence']
            diff_js = res['js_divergence'] - full['js_divergence']
            diff_corr = res['correlation'] - full['correlation']
            diff_attn = full['attention_accuracy'] - res['attention_accuracy']
            diff_mae = res['duration_mae'] - full['duration_mae']
            print(f"{name:<20} {res['top1_accuracy']:>10.2%} ({diff_top1:+.2%}) "
                  f"{res['top3_accuracy']:>10.2%} ({diff_top3:+.2%}) "
                  f"{res['kl_divergence']:>10.4f} ({diff_kl:+.4f}) "
                  f"{res['js_divergence']:>10.4f} ({diff_js:+.4f}) "
                  f"{res['correlation']:>10.4f} ({diff_corr:+.4f}) "
                  f"{res['attention_accuracy']:>10.2%} ({diff_attn:+.2%}) "
                  f"{res['duration_mae']:>10.1f}s ({diff_mae:+.1f})")

    def _print_latex_table(self, results: Dict):
        """打印LaTeX表格"""
        print("\n" + "="*90)
        print("LaTeX Table for Ablation Study")
        print("="*90)

        print("\n\\begin{table}[t]")
        print("\\centering")
        print("\\caption{Ablation study results. We report both traditional Top-K accuracy and distribution matching metrics.}")
        print("\\label{tab:ablation}")
        print("\\begin{tabular}{lccccccc}")
        print("\\hline")
        print("Variant & Top-1$\\uparrow$ & Top-3$\\uparrow$ & KL$\\downarrow$ & JS$\\downarrow$ & Corr$\\uparrow$ & Attn$\\uparrow$ & MAE$\\downarrow$ \\\\")
        print("\\hline")

        full = results.get('Full', {})
        print(f"Full (Ours) & {full['top1_accuracy']:.1%} & {full['top3_accuracy']:.1%} & "
              f"{full['kl_divergence']:.4f} & {full['js_divergence']:.4f} & "
              f"{full['correlation']:.4f} & {full['attention_accuracy']:.2%} & {full['duration_mae']:.1f}s \\\\")

        for name, res in results.items():
            if name == 'Full':
                continue
            diff_top1 = full['top1_accuracy'] - res['top1_accuracy']
            diff_top3 = full['top3_accuracy'] - res['top3_accuracy']
            diff_kl = res['kl_divergence'] - full['kl_divergence']
            diff_js = res['js_divergence'] - full['js_divergence']
            diff_corr = res['correlation'] - full['correlation']
            diff_attn = full['attention_accuracy'] - res['attention_accuracy']
            diff_mae = res['duration_mae'] - full['duration_mae']
            print(f"w/o {name} & {res['top1_accuracy']:.1%} ({diff_top1:+.1%}) & "
                  f"{res['top3_accuracy']:.1%} ({diff_top3:+.1%}) & "
                  f"{res['kl_divergence']:.4f} ({diff_kl:+.4f}) & "
                  f"{res['js_divergence']:.4f} ({diff_js:+.4f}) & "
                  f"{res['correlation']:.4f} ({diff_corr:+.4f}) & "
                  f"{res['attention_accuracy']:.2%} ({diff_attn:+.2%}) & "
                  f"{res['duration_mae']:.1f}s ({diff_mae:+.1f}) \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="分布匹配度消融实验")
    parser.add_argument("--model", default="Qwen")
    parser.add_argument("--api-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    experiment = DistributionAblation(
        model_name=args.model,
        api_url=args.api_url,
        data_path=args.data
    )
    experiment.run_ablation_study()


if __name__ == "__main__":
    main()
