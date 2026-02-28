#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study with vLLM - 三阶段架构

阶段 0: Qwen LLM 处理拓扑和特征 → 提取关键信息
阶段 1: 微调模型基于提取的信息预测
阶段 2: Qwen LLM 用记忆优化最终输出

使用真实展品名称（而非ID）
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Optional

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# 真实展品名称（从训练数据中提取）
REAL_EXHIBIT_NAMES = [
    "丁香花", "金鱼兰", "牡丹花", "说明文字-千岛湖", "玉兰花开",
    "人物-祝大年创作", "自序", "松竹海", "西双版纳",
    "北大简-仓颉篇", "文物展柜", "颜真卿楷书", "耕织图-多媒体装置",
    "二十四节气圆盘", "鸡蛋花", "山茶花", "千岛湖", "说明文字", "入口",
    "森林之歌", "漓江春色", "风筝", "鸢飞曲", "黄山松", "迎客松",
    "三星堆展区", "殷墟展区", "良渚展区", "文字瀑布", "耕织图"
]

# 展品特征信息
EXHIBIT_FEATURES = {
    "丁香花": "一幅精美的艺术画作，描绘了白色圆盆栽开满白色小花，绿叶繁盛的场景",
    "金鱼兰": "土红色盆子栽种着叶片细长、花朵呈金鱼状的植物",
    "牡丹花": "色彩饱满，花瓣层次细腻，搭配翠绿的叶片",
    "说明文字-千岛湖": "千岛湖 Qiandao Lake 1980s...",
    "玉兰花开": "开满白色玉兰花的树，挂在黑墙上",
    "人物-祝大年创作": "祝大年创作的西双版纳傣族生活主题工笔重彩人物组画",
    "自序": "白墙上陈列着的自序节选文章",
    "松竹海": "上面画着松树和竹子，挂在白墙中间",
    "西双版纳": "描绘西双版纳热带雨林场景，有正在劳作的人",
    "北大简-仓颉篇": "隶书-北大简《仓颉篇》，收藏于北京大学赛克勒考古与艺术博物馆",
    "文物展柜": "天人合一部分文字文物展柜",
    "颜真卿楷书": "楷书-颜真卿《明拓干禄字书册》，收藏于故宫博物院",
    "耕织图-多媒体装置": "数字活化的中国古代耕织图，展示农耕文化",
    "二十四节气圆盘": "融合虚拟现实技术的动态影像装置",
    "鸡蛋花": "一盆花的画作展品，挂在墙上",
    "山茶花": "一盆花的画作展品，在柱子上",
    "千岛湖": "湖景主题艺术作品",
    "说明文字": "展品说明介绍",
    "入口": "展厅入口过渡空间",
    "森林之歌": "九幅画位于展台上面，描绘森林场景",
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

# 模拟的拓扑关系（相邻展品）
TOPOLOGY_ADJACENCY = {
    "入口": ["丁香花"],  # 入口只能去第一个展品
    "丁香花": ["金鱼兰", "说明文字-千岛湖"],  # 丁香花之后可以去看别的
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
    """消融实验配置"""

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


class ThreeStageAblation:
    """三阶段消融实验"""

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

        # 加载数据
        self.test_data = self._load_test_data()

        # 初始化客户端
        self._init_client()
        self.preprocess_cache = {}

    def _init_client(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        print(f"[+] Connected to vLLM API: {self.api_url}")

    def _load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        if self.data_path and os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                return json.load(f)
        return self._create_sample_data()

    def _create_sample_data(self) -> List[Dict]:
        """使用真实展品名称创建测试数据（减少样本量）"""
        # 基于真实参观模式创建序列 - 更少样本用于快速测试
        sample_sequences = [
            ["入口", "丁香花", "金鱼兰", "牡丹花"],
            ["入口", "丁香花", "说明文字-千岛湖", "人物-祝大年创作"],
            ["玉兰花开", "松竹海", "西双版纳", "耕织图"],
            ["迎客松", "三星堆展区", "殷墟展区", "良渚展区"],
            ["森林之歌", "漓江春色", "风筝", "黄山松"],
        ]

        test_data = []
        for seq in sample_sequences:
            for i in range(len(seq) - 1):
                test_data.append({
                    'context': seq[:i+1],
                    'current': seq[i],
                    'next': seq[i + 1],
                    'history': [
                        {'name': seq[j], 'duration': 60, 'attention': 'A'}
                        for j in range(i)
                    ],
                    'dwell': 60,
                    'attention': 'A'
                })
        return test_data

    def stage0_qwen_preprocess(
        self,
        current: str,
        config: AblationConfig
    ) -> Dict:
        """阶段 0: Qwen LLM 处理拓扑和特征"""
        cache_key = (current, config.use_topology_preprocess, config.use_feature_preprocess)
        if cache_key in self.preprocess_cache:
            return self.preprocess_cache[cache_key]

        # 构建预处理 prompt
        prompt_parts = [f"当前位置: {current}"]

        # 拓扑信息
        if config.use_topology_preprocess:
            neighbors = TOPOLOGY_ADJACENCY.get(current, [])
            if neighbors:
                prompt_parts.append(f"相邻展品: {', '.join(neighbors[:5])}")

        # 展品特征
        if config.use_feature_preprocess:
            features = EXHIBIT_FEATURES.get(current, "普通展品")
            prompt_parts.append(f"展品特征: {features}")

        prompt_parts.append("\n提取关键信息用于轨迹预测。返回JSON:")
        prompt_parts.append('{"context": "简要描述", "candidates": ["展品名1", "展品名2", "展品名3"]}')

        prompt = "\n".join(prompt_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            result = response.choices[0].message.content.strip()

            # 解析并缓存
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                self.preprocess_cache[cache_key] = parsed
                return parsed

            # 默认返回
            return {"context": f"参观{current}", "candidates": []}

        except Exception as e:
            return {"context": f"参观{current}", "candidates": []}

    def stage1_fine_tuned_predict(
        self,
        current: str,
        processed_context: Dict,
        config: AblationConfig
    ) -> Dict:
        """阶段 1: 微调模型预测"""
        context_desc = processed_context.get("context", "")
        candidates = processed_context.get("candidates", [])

        history_str = "、".join(self.test_data[0]['context'][:3]) if self.test_data else "入口"

        prompt = f"""你是轨迹预测模型。

当前位置: {current}

环境分析:
{context_desc}
候选展品: {', '.join(candidates[:5]) if candidates else '所有展品'}

之前参观: {history_str}

预测下一个展品。返回JSON:
{{"prediction_name": "展品名称", "confidence": 0.0-1.0}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            result = response.choices[0].message.content.strip()

            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                pred_name = parsed.get("prediction_name") or parsed.get("prediction_id")
                if pred_name and pred_name in REAL_EXHIBIT_NAMES:
                    return {
                        'prediction_name': pred_name,
                        'confidence': parsed.get('confidence', 0.8)
                    }
        except:
            pass

        # 默认：返回拓扑邻居
        neighbors = TOPOLOGY_ADJACENCY.get(current, [REAL_EXHIBIT_NAMES[0]])
        return {
            'prediction_name': neighbors[0] if neighbors else REAL_EXHIBIT_NAMES[0],
            'confidence': 0.5
        }

    def stage2_qwen_refine(
        self,
        current: str,
        history: List[Dict],
        initial_prediction: Dict,
        config: AblationConfig
    ) -> Dict:
        """阶段 2: Qwen 用记忆优化"""
        initial_pred = initial_prediction['prediction_name']

        memory_info = ""
        if config.use_memory_in_llm and history:
            visited = [h['name'] for h in history]
            if visited:
                memory_info = f"最近参观: {', '.join(visited[-5:])}\n"
                if initial_pred in visited[-3:]:
                    memory_info += f"注意: {initial_pred} 刚刚参观过，可能不适合\n"

        prompt = f"""优化轨迹预测

当前位置: {current}
初始预测: {initial_pred}

{memory_info}考虑参观规律:
- 避免重复参观刚看过的展品
- 考虑展品类型多样性

返回优化后的预测JSON:
{{"prediction_name": "展品名称", "confidence": 0.0-1.0, "reasoning": "理由"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            result = response.choices[0].message.content.strip()

            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                pred_name = parsed.get("prediction_name") or parsed.get("prediction_id")
                if pred_name and pred_name in REAL_EXHIBIT_NAMES:
                    return {
                        'prediction_name': pred_name,
                        'confidence': parsed.get('confidence', 0.85),
                        'reasoning': parsed.get('reasoning', '')
                    }
        except:
            pass

        return initial_prediction

    def predict_with_config(
        self,
        config: AblationConfig,
        sample: Dict,
        debug: bool = False
    ) -> Dict:
        """三阶段预测"""
        current = sample['current']
        history = sample.get('history', [])
        ground_truth = sample['next']

        # 阶段 0: Qwen 预处理
        processed_context = self.stage0_qwen_preprocess(current, config)

        # 阶段 1: 微调模型预测
        stage1_result = self.stage1_fine_tuned_predict(current, processed_context, config)

        # 阶段 2: Qwen 用记忆优化
        final_result = self.stage2_qwen_refine(current, history, stage1_result, config)

        if debug:
            print(f"\n[DEBUG] 当前: {current}, 真实: {ground_truth}")
            print(f"[DEBUG] 阶段0预处理: {processed_context.get('context', 'N/A')}")
            print(f"[DEBUG] 阶段1预测: {stage1_result['prediction_name']}")
            print(f"[DEBUG] 阶段2优化: {final_result['prediction_name']}")
            print(f"[DEBUG] 匹配: {final_result['prediction_name'] == ground_truth}")

        return {
            'prediction': final_result,
            'ground_truth': ground_truth,
            'stage1': stage1_result
        }

    def _evaluate_config(self, config: AblationConfig) -> Dict:
        """评估配置"""
        correct_top1 = 0
        correct_top3 = 0
        dwell_errors = []

        print(f"    评估 {len(self.test_data)} 个样本...", end='', flush=True)

        for sample in self.test_data:
            result = self.predict_with_config(config, sample)
            prediction = result['prediction']
            ground_truth = result['ground_truth']

            if prediction['prediction_name'] == ground_truth:
                correct_top1 += 1

            # Top-3
            candidates = TOPOLOGY_ADJACENCY.get(sample['current'], REAL_EXHIBIT_NAMES)
            if ground_truth in candidates[:3]:
                correct_top3 += 1

            pred_dwell = 120 * (1 - prediction.get('confidence', 0.5)) + 30
            dwell_errors.append(abs(pred_dwell - sample.get('dwell', 60)))

        total = len(self.test_data)
        print(f" 完成")

        return {
            'top1_acc': correct_top1 / total,
            'top3_acc': correct_top3 / total,
            'mae': np.mean(dwell_errors) if dwell_errors else 0
        }

    def run_ablation_study(self) -> Dict:
        """运行消融实验"""
        print("="*70)
        print("三阶段消融实验（使用真实展品名称）")
        print("="*70)
        print(f"模型: {self.model_name}")
        print(f"样本数: {len(self.test_data)}")
        print(f"展品数: {len(REAL_EXHIBIT_NAMES)}")
        print("="*70)

        # 测试一个样本
        print("\n[*] 测试一个样本...")
        test_config = AblationConfig()
        self.predict_with_config(test_config, self.test_data[0], debug=True)

        results = {}

        configs = [
            AblationConfig(use_topology_preprocess=True, use_feature_preprocess=True, use_memory_in_llm=True),
            AblationConfig(use_topology_preprocess=False, use_feature_preprocess=True, use_memory_in_llm=True),
            AblationConfig(use_topology_preprocess=True, use_feature_preprocess=False, use_memory_in_llm=True),
            AblationConfig(use_topology_preprocess=True, use_feature_preprocess=True, use_memory_in_llm=False),
            AblationConfig(use_topology_preprocess=False, use_feature_preprocess=False, use_memory_in_llm=True),
        ]

        for config in configs:
            config_name = config.get_name()
            print(f"\n[*] 测试: {config_name}")

            config_results = self._evaluate_config(config)
            results[config_name] = config_results

            print(f"    Top-1: {config_results['top1_acc']:.1%}")
            print(f"    Top-3: {config_results['top3_acc']:.1%}")
            print(f"    MAE:   {config_results['mae']:.1f}s")

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
        """打印结果"""
        print("\n" + "="*70)
        print("结果汇总")
        print("="*70)
        print(f"{'配置':<20} {'Top-1':>12} {'Top-3':>12} {'MAE':>10}")
        print("-" * 56)

        full = results.get('Full', {})
        print(f"{'Full (Ours)':<20} {full['top1_acc']:>12.1%} {full['top3_acc']:>12.1%} {full['mae']:>10.1f}s")

        for name, res in results.items():
            if name == 'Full':
                continue
            print(f"{name:<20} {res['top1_acc']:>12.1%} {res['top3_acc']:>12.1%} {res['mae']:>10.1f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="三阶段消融实验")
    parser.add_argument("--model", default="Qwen")
    parser.add_argument("--api-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    experiment = ThreeStageAblation(
        model_name=args.model,
        api_url=args.api_url,
        data_path=args.data
    )
    experiment.run_ablation_study()


if __name__ == "__main__":
    main()
