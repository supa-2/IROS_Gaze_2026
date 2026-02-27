#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study - 消融实验

测试系统各个组件的贡献
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class AblationConfig:
    """消融实验配置"""

    def __init__(
        self,
        use_memory: bool = True,
        use_topology: bool = True,
        use_feature_extractor: bool = True,
        use_multi_step: bool = True,
        n_steps: int = 5
    ):
        self.use_memory = use_memory
        self.use_topology = use_topology
        self.use_feature_extractor = use_feature_extractor
        self.use_multi_step = use_multi_step
        self.n_steps = n_steps if use_multi_step else 1

    def get_name(self) -> str:
        """获取配置名称"""
        parts = []
        if not self.use_memory:
            parts.append("No-Memory")
        if not self.use_topology:
            parts.append("No-Topology")
        if not self.use_feature_extractor:
            parts.append("No-Extractor")
        if not self.use_multi_step:
            parts.append("No-Multi-step")
        return "-".join(parts) if parts else "Full"


class AblationPredictor:
    """
    消融实验预测器

    根据不同配置运行预测，评估各组件贡献
    """

    def __init__(self, config: AblationConfig):
        self.config = config

        # 导入必要模块
        from skills.topology.graph_engine import TopologyEngine
        from skills.memory.manager import MemoryManager
        from skills.prediction.context_builder import ContextBuilder
        from skills.prediction.llm_reasoner import LLMReasoner
        from skills.prediction.sequence_predictor import SequencePredictor
        from config import Config, AttentionConfig

        self.TopologyEngine = TopologyEngine
        self.MemoryManager = MemoryManager
        self.ContextBuilder = ContextBuilder
        self.LLMReasoner = LLMReasoner
        self.SequencePredictor = SequencePredictor

        self.config_cls = Config
        self.AttentionConfig = AttentionConfig

    def predict_single(
        self,
        current_exhibit: str,
        history: List[Dict],
        topology_engine,
        all_exhibits: List[str] = None
    ) -> Dict:
        """
        单次预测

        Args:
            current_exhibit: 当前展品ID
            history: 历史记录 [{'id':, 'name':, 'level':, 'duration':}]
            topology_engine: 拓扑引擎
            all_exhibits: 所有展品列表（用于No-Topology）

        Returns:
            预测结果
        """
        # 1. 构建上下文
        context = self._build_context(current_exhibit, history, topology_engine, all_exhibits)

        # 2. LLM推理
        attention_config = self.AttentionConfig()
        config = self.config_cls()

        reasoner = self.LLMReasoner(config.model)
        prediction = reasoner.predict_next(context, attention_config)

        return prediction

    def _build_context(
        self,
        current_exhibit: str,
        history: List[Dict],
        topology_engine,
        all_exhibits: List[str]
    ) -> Dict:
        """根据配置构建上下文"""
        # 获取当前展品信息
        current_info = topology_engine.query_node(current_exhibit)

        # 处理历史（No-Memory: 清空历史）
        if not self.config.use_memory:
            history = []

        # 格式化历史
        formatted_history = []
        for h in history[-5:]:  # 最多保留5条
            formatted_history.append({
                'id': h.get('id'),
                'name': h.get('name', h.get('id', '')),
                'level': h.get('level', 'C'),
                'duration': h.get('duration', 30)
            })

        # 构建空间上下文
        spatial_context = current_info.get('context', {})

        # No-Topology: 使用所有展品作为候选
        if not self.config.use_topology and all_exhibits:
            spatial_context['direct_choices'] = [
                {'id': eid, 'relation': 'candidate'}
                for eid in all_exhibits if eid != current_exhibit
            ]

        # No-Feature Extractor: 使用简化的展品信息
        if not self.config.use_feature_extractor:
            current_info['info']['features'] = ''
            for choice in spatial_context.get('direct_choices', []):
                choice['name'] = choice.get('id', '')

        # 组装上下文
        context = {
            'current': {
                'id': current_exhibit,
                'name': current_info['info']['name'],
                'features': current_info['info'].get('features', ''),
                'attention_level': 'C',
                'estimated_duration': 30
            },
            'history': formatted_history,
            'spatial': {
                'previous_path': spatial_context.get('previous_path', []),
                'next_path': spatial_context.get('next_path', []),
                'reachable_options': spatial_context.get('direct_choices', [])
            },
            'statistics': {
                'total_gazes': len(history),
                'unique_exhibits': len(set(h.get('id') for h in history)),
                'visited_exhibits': list(set(h.get('id') for h in history))
            }
        }

        return context


class AblationExperiment:
    """消融实验运行器"""

    def __init__(self, map_name: str = 'TH', data_path: str = None):
        self.map_name = map_name
        self.data_path = data_path

        # 初始化拓扑引擎
        from skills.topology.graph_engine import TopologyEngine
        self.topology = TopologyEngine(map_name)
        self.all_exhibits = list(self.topology.graph.nodes())

        # 加载测试数据
        self.test_data = self._load_test_data()

    def _load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        if self.data_path and os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                return json.load(f)

        # 如果没有数据，创建模拟数据
        print("[!] No test data found, creating sample data...")
        return self._create_sample_data()

    def _create_sample_data(self) -> List[Dict]:
        """创建模拟测试数据"""
        # 生成一些测试序列
        sample_sequences = [
            ['TH-E01', 'TH-I-B01', 'TH-B02', 'TH-C03', 'TH-D04'],
            ['TH-E01', 'TH-B02', 'TH-C03', 'TH-E05', 'TH-F06'],
            ['TH-A01', 'TH-B02', 'TH-D04', 'TH-E01', 'TH-I-B01'],
            ['TH-C03', 'TH-D04', 'TH-E05', 'TH-F06', 'TH-G07'],
            ['TH-B02', 'TH-C03', 'TH-D04', 'TH-E05', 'TH-F06'],
        ]

        test_data = []
        for seq in sample_sequences:
            for i in range(len(seq) - 1):
                test_data.append({
                    'current': seq[i],
                    'next': seq[i + 1],
                    'history': [{'id': seq[j], 'level': 'A', 'duration': 60} for j in range(i)],
                    'dwell': 60,
                    'attention': 'A'
                })

        return test_data[:20]  # 返回20条测试数据

    def run_ablation_study(self) -> Dict:
        """运行完整消融实验"""
        print("="*60)
        print("Ablation Study Experiment")
        print("="*60)

        results = {}

        # 定义所有消融配置
        configs = [
            # Full model
            AblationConfig(use_memory=True, use_topology=True,
                         use_feature_extractor=True, use_multi_step=True),
            # No Memory
            AblationConfig(use_memory=False, use_topology=True,
                         use_feature_extractor=True, use_multi_step=True),
            # No Topology
            AblationConfig(use_memory=True, use_topology=False,
                         use_feature_extractor=True, use_multi_step=True),
            # No Feature Extractor
            AblationConfig(use_memory=True, use_topology=True,
                         use_feature_extractor=False, use_multi_step=True),
            # No Multi-step (K=1)
            AblationConfig(use_memory=True, use_topology=True,
                         use_feature_extractor=True, use_multi_step=False),
        ]

        for config in configs:
            config_name = config.get_name()
            print(f"\n[*] Testing: {config_name}")

            predictor = AblationPredictor(config)
            config_results = self._evaluate_config(predictor)

            results[config_name] = config_results

            print(f"    Top-1: {config_results['top1_acc']:.1%}")
            print(f"    MAE:   {config_results['mae']:.1f}s")
            print(f"    Attn:  {config_results['attn_acc']:.1%}")
            print(f"    Regret: {config_results['regret']:.2f}")

        # 保存结果
        self._save_results(results)

        # 打印对比表格
        self._print_comparison_table(results)

        return results

    def _evaluate_config(self, predictor: AblationPredictor) -> Dict:
        """评估单个配置"""
        correct_top1 = 0
        correct_top3 = 0
        dwell_errors = []
        attention_correct = 0

        for sample in self.test_data:
            current = sample['current']
            ground_truth = sample['next']
            history = sample.get('history', [])

            # 预测
            prediction = predictor.predict_single(
                current, history, self.topology,
                self.all_exhibits if not predictor.config.use_topology else None
            )

            # Top-1 准确率
            if prediction.get('prediction_id') == ground_truth:
                correct_top1 += 1

            # Top-3 准确率（简化：假设正确的在top3里）
            # 实际需要从模型获取top3
            if prediction.get('prediction_id') == ground_truth:
                correct_top3 += 1

            # Dwell MAE
            predicted_dwell = prediction.get('estimated_duration', 30)
            true_dwell = sample.get('dwell', 60)
            dwell_errors.append(abs(predicted_dwell - true_dwell))

            # Attention 准确率
            predicted_attn = prediction.get('attention_level', 'C')
            true_attn = sample.get('attention', 'C')
            if predicted_attn == true_attn:
                attention_correct += 1

        # Robot Regret（简化：用预测错误率）
        total = len(self.test_data)
        regret = (total - correct_top1) / max(total, 1) * 2  # 转换为步数

        return {
            'top1_acc': correct_top1 / max(total, 1),
            'top3_acc': correct_top3 / max(total, 1),
            'mae': np.mean(dwell_errors) if dwell_errors else 0,
            'attn_acc': attention_correct / max(total, 1),
            'regret': regret
        }

    def _save_results(self, results: Dict):
        """保存结果"""
        output_dir = "data/outputs/ablation"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "ablation_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[+] Results saved to {output_path}")

    def _print_comparison_table(self, results: Dict):
        """打印对比表格（LaTeX格式）"""
        print("\n" + "="*60)
        print("LaTeX Table for Ablation Study")
        print("="*60)

        print("\n\\begin{table}[t]")
        print("\\centering")
        print("\\caption{Ablation Study}")
        print("\\label{tab:ablation}")
        print("\\begin{tabular}{lcccc}")
        print("\\hline")
        print("Variant & Top-1 $\\uparrow$ & MAE$\\downarrow$ & Attn $\\uparrow$ & Regret$\\downarrow$ \\\\")
        print("\\hline")

        # Full (Ours) 放在最前面
        full = results.get('Full', {})
        print(f"Full (Ours) & {full['top1_acc']:.1%} & {full['mae']:.1f}s & {full['attn_acc']:.1%} & {full['regret']:.1f} \\\\")

        # 其他变体
        for name, res in results.items():
            if name == 'Full':
                continue
            # 计算差异
            diff_top1 = (full['top1_acc'] - res['top1_acc']) * 100
            diff_mae = res['mae'] - full['mae']
            diff_attn = (full['attn_acc'] - res['attn_acc']) * 100
            diff_regret = res['regret'] - full['regret']

            print(f"-{name} & {res['top1_acc']:.1%} ({diff_top1:+.1f}) & "
                  f"{res['mae']:.1f}s ({diff_mae:+.1f}) & "
                  f"{res['attn_acc']:.1%} ({diff_attn:+.1f}) & "
                  f"{res['regret']:.1f} ({diff_regret:+.1f}) \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="TH")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    experiment = AblationExperiment(args.map, args.data)
    results = experiment.run_ablation_study()
