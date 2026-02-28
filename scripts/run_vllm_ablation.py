#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study with Distribution Matching Evaluation

使用分布匹配度评估，而非单一正确答案
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Optional
from collections import Counter, defaultdict
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

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
        """创建测试数据"""
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
        test_data = []
        for seq in sample_sequences:
            for i in range(len(seq) - 1):
                test_data.append({
                    'current': seq[i],
                    'next': seq[i + 1],
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
        """获取模型预测的分布"""
        # 获取拓扑和特征信息
        neighbors = TOPOLOGY_ADJACENCY.get(current, [])
        features = EXHIBIT_FEATURES.get(current, "")

        # 构建prompt，要求模型返回概率分布
        prompt_parts = [
            f"当前位置: {current}",
        ]

        if config.use_topology_preprocess and neighbors:
            prompt_parts.append(f"相邻展品: {', '.join(neighbors)}")

        if config.use_feature_preprocess:
            prompt_parts.append(f"展品特征: {features}")

        prompt_parts.append(f"""
基于上述信息，预测从 {current} 出发，游客选择各个相邻展品的概率分布。

候选展品: {', '.join(candidates)}

返回JSON格式，包含每个候选展品的预测概率（概率和为1）:
{{"predictions": [{{"name": "展品1", "probability": 0.5}}, {{"name": "展品2", "probability": 0.3}}, ...]}}
""")

        prompt = "\n".join(prompt_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            result = response.choices[0].message.content.strip()

            # 解析
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
                        if name and name in candidates:
                            dist[name] = prob
                    # 归一化
                    total = sum(dist.values())
                    if total > 0:
                        dist = {k: v/total for k, v in dist.items()}
                    return dist
        except Exception as e:
            print(f"[!] Prediction error: {e}")

        # 默认：均匀分布
        return {c: 1.0/len(candidates) for c in candidates}

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

        # 对每个有真实分布的起点进行评估
        for current, real_dist in self.real_distributions.items():
            candidates = list(real_dist.keys())

            # 获取模型预测的分布
            model_dist = self.get_model_distribution(current, candidates, config)

            # 计算指标
            try:
                kl = self.kl_divergence(real_dist, model_dist)
                js = self.js_divergence(real_dist, model_dist)
                corr = self.correlation(real_dist, model_dist)

                kl_divs.append(kl)
                js_divs.append(js)
                corrs.append(corr)
            except:
                continue

        print(f" 完成 ({len(kl_divs)} 个起点)")

        return {
            'kl_divergence': np.mean(kl_divs) if kl_divs else 0,
            'js_divergence': np.mean(js_divs) if js_divs else 0,
            'correlation': np.mean(corrs) if corrs else 0,
            'num_evaluated': len(kl_divs)
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

            print(f"    KL散度: {config_results['kl_divergence']:.4f} ↓ (越低越好)")
            print(f"    JS散度: {config_results['js_divergence']:.4f} ↓ (越低越好)")
            print(f"    相关系数: {config_results['correlation']:.4f} ↑ (越高越好)")

        self._save_results(results)
        self._print_table(results)
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
        print("\n" + "="*70)
        print("分布匹配度结果")
        print("="*70)
        print(f"{'配置':<20} {'KL散度↓':>12} {'JS散度↓':>12} {'相关系数↑':>12}")
        print("-" * 58)

        full = results.get('Full', {})
        print(f"{'Full (Ours)':<20} {full['kl_divergence']:>12.4f} {full['js_divergence']:>12.4f} {full['correlation']:>12.4f}")

        for name, res in results.items():
            if name == 'Full':
                diff_kl = res['kl_divergence'] - full['kl_divergence']
                diff_js = res['js_divergence'] - full['js_divergence']
                diff_corr = res['correlation'] - full['correlation']
                print(f"{name:<20} {res['kl_divergence']:>12.4f} ({diff_kl:+.4f}) "
                      f"{res['js_divergence']:>12.4f} ({diff_js:+.4f}) "
                      f"{res['correlation']:>12.4f} ({diff_corr:+.4f})")


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
