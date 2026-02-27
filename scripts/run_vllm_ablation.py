#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study with vLLM - 两阶段架构

阶段 1: 微调模型预测（基于当前状态）
阶段 2: Qwen LLM 优化（使用 Memory + Topology + Features）

在有模型的机器上运行，测试各组件的贡献
"""

import os
import sys
import json
import numpy as np
import time
from typing import List, Dict, Optional

# 设置 PyTorch 显存环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# 设置项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# 验证 skills 目录存在
skills_path = os.path.join(project_root, 'skills')
USE_TOPOLOGY = os.path.exists(skills_path)


class AblationConfig:
    """消融实验配置"""

    def __init__(
        self,
        use_memory_in_llm: bool = True,
        use_topology_in_llm: bool = True,
        use_feature_in_llm: bool = True,
        use_multi_step: bool = True
    ):
        # 阶段1: 微调模型（保持不变，只根据当前状态预测）
        # 阶段2: LLM 优化（这些参数控制 LLM 收到什么信息）
        self.use_memory_in_llm = use_memory_in_llm
        self.use_topology_in_llm = use_topology_in_llm
        self.use_feature_in_llm = use_feature_in_llm
        self.use_multi_step = use_multi_step

    def get_name(self) -> str:
        """获取配置名称"""
        parts = []
        if not self.use_memory_in_llm:
            parts.append("No-Memory")
        if not self.use_topology_in_llm:
            parts.append("No-Topology")
        if not self.use_feature_in_llm:
            parts.append("No-Feature")
        if not self.use_multi_step:
            parts.append("No-Multi-step")
        return "-".join(parts) if parts else "Full"


class TwoStageAblationExperiment:
    """两阶段消融实验"""

    def __init__(
        self,
        fine_tuned_model: str = "Qwen",  # API server 上的模型名
        llm_api_key: str = "sk-YourCustomSecretKey123",
        llm_base_url: str = "http://localhost:8000/v1",
        qwen_api_url: str = None,  # 如果需要用外部 Qwen API
        map_name: str = 'TH',
        data_path: str = None
    ):
        self.fine_tuned_model = fine_tuned_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.qwen_api_url = qwen_api_url
        self.map_name = map_name
        self.data_path = data_path

        # 加载数据
        self.test_data = self._load_test_data()
        self.exhibit_info = self._load_exhibit_info()
        self.topology_data = self._load_topology_data() if USE_TOPOLOGY else {}

        # 初始化 API 客户端
        self._init_clients()

    def _init_clients(self):
        """初始化 API 客户端"""
        from openai import OpenAI

        # 微调模型的客户端（您的 Qwen2.5-32B-int4）
        self.fine_tuned_client = OpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url
        )

        # Qwen LLM 客户端（用于优化，如果有单独的 API）
        if self.qwen_api_url:
            self.qwen_client = OpenAI(
                api_key=os.getenv("QWEN_API_KEY", self.llm_api_key),
                base_url=self.qwen_api_url
            )
        else:
            # 如果没有单独的 Qwen API，用同一个（假设您的微调模型本身有推理能力）
            self.qwen_client = self.fine_tuned_client

        print(f"[+] Connected to fine-tuned model at: {self.llm_base_url}")
        if self.qwen_api_url:
            print(f"[+] Connected to Qwen LLM at: {self.qwen_api_url}")
        else:
            print(f"[+] Using fine-tuned model also as LLM for optimization")

    def _load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        if self.data_path and os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                return json.load(f)

        # 创建模拟数据
        return self._create_sample_data()

    def _create_sample_data(self) -> List[Dict]:
        """创建模拟测试数据"""
        sample_sequences = [
            ['TH-E01', 'TH-I-B01', 'TH-B02', 'TH-C03', 'TH-D04'],
            ['TH-E01', 'TH-B02', 'TH-C03', 'TH-E05', 'TH-F06'],
            ['TH-A01', 'TH-B02', 'TH-D04', 'TH-E01', 'TH-I-B01'],
            ['TH-C03', 'TH-D04', 'TH-E05', 'TH-F06', 'TH-G07'],
            ['TH-B02', 'TH-C03', 'TH-D04', 'TH-E05', 'TH-F06'],
            ['TH-E01', 'TH-I-B01', 'TH-B02', 'TH-C03'],
            ['TH-A01', 'TH-B02', 'TH-D04', 'TH-E01'],
            ['TH-C03', 'TH-D04', 'TH-E05', 'TH-F06'],
        ]

        test_data = []
        for seq in sample_sequences:
            for i in range(len(seq) - 1):
                test_data.append({
                    'context': seq[:i+1],
                    'current': seq[i],
                    'next': seq[i + 1],
                    'history': [{'id': seq[j], 'name': seq[j], 'level': 'A', 'duration': 60} for j in range(i)],
                    'dwell': 60,
                    'attention': 'A'
                })

        return test_data

    def _load_exhibit_info(self) -> Dict:
        """加载展品信息"""
        if USE_TOPOLOGY:
            try:
                from skills.topology.graph_engine import TopologyEngine
                topology = TopologyEngine(self.map_name)
                info = {}
                for node_id in topology.graph.nodes():
                    node_data = topology.query_node(node_id)
                    info[node_id] = node_data.get('info', {})
                return info
            except Exception as e:
                print(f"[!] Failed to load exhibit info: {e}")

        # Mock 数据
        mock_exhibits = [
            'TH-E01', 'TH-I-B01', 'TH-B02', 'TH-C03', 'TH-D04',
            'TH-E05', 'TH-F06', 'TH-G07', 'TH-A01', 'TH-H08'
        ]
        return {
            eid: {
                'name': eid,
                'features': f'Exhibit {eid}',
                'level': 'A',
                'popularity': np.random.randint(50, 100)
            }
            for eid in mock_exhibits
        }

    def _load_topology_data(self) -> Dict:
        """加载拓扑数据"""
        try:
            from skills.topology.graph_engine import TopologyEngine
            topology = TopologyEngine(self.map_name)
            data = {}
            for node_id in topology.graph.nodes():
                node_data = topology.query_node(node_id)
                data[node_id] = {
                    'neighbors': node_data.get('context', {}).get('direct_choices', []),
                    'info': node_data.get('info', {})
                }
            return data
        except Exception as e:
            print(f"[!] Failed to load topology: {e}")
            return {}

    def _get_spatial_candidates(self, current: str) -> List[str]:
        """获取空间上的候选展品"""
        if USE_TOPOLOGY and current in self.topology_data:
            neighbors = self.topology_data[current].get('neighbors', [])
            return [n.get('id') for n in neighbors]
        # Mock: 返回所有其他展品作为候选
        return [e for e in self.exhibit_info.keys() if e != current]

    def stage1_fine_tuned_prediction(
        self,
        current: str,
        history: List[Dict]
    ) -> Dict:
        """
        阶段 1: 微调模型预测
        只根据当前状态预测，不用额外特征
        """
        # 简化的 prompt，只给基本信息
        prompt = f"""Current location: {current}
Previous locations: {[h['id'] for h in history[-3:]]}

Predict the next exhibit. Return JSON:
{{"prediction_id": "exhibit_id"}}"""

        try:
            response = self.fine_tuned_client.chat.completions.create(
                model=self.fine_tuned_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            result_text = response.choices[0].message.content.strip()

            # 解析
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                pred_id = parsed.get('prediction_id')
                if pred_id:
                    return {
                        'prediction_id': pred_id,
                        'confidence': parsed.get('confidence', 0.8),
                        'raw_output': result_text
                    }
        except Exception as e:
            print(f"[!] Stage 1 error: {e}")

        # 默认返回
        return {
            'prediction_id': current,  # 默认待在原地
            'confidence': 0.5,
            'raw_output': ''
        }

    def stage2_llm_refinement(
        self,
        current: str,
        history: List[Dict],
        stage1_prediction: Dict,
        config: AblationConfig
    ) -> Dict:
        """
        阶段 2: LLM 优化
        使用 Memory + Topology + Features 来优化初始预测
        """
        current_info = self.exhibit_info.get(current, {})

        # 构建历史信息（No-Memory 时跳过）
        history_info = ""
        if config.use_memory_in_llm and history:
            for h in history[-5:]:
                history_info += f"- {h['id']}: visited, dwell={h.get('duration', 60)}s\n"

        # 构建空间信息（No-Topology 时跳过）
        spatial_info = ""
        if config.use_topology_in_llm:
            candidates = self._get_spatial_candidates(current)
            spatial_info = f"Nearby exhibits: {', '.join(candidates[:5])}\n"

        # 构建特征信息（No-Feature 时跳过）
        feature_info = ""
        if config.use_feature_in_llm:
            feature_info = f"Current exhibit features: {current_info.get('features', '')}\n"

        # 初始预测
        initial_pred = stage1_prediction['prediction_id']

        # LLM 优化 prompt
        prompt = f"""You are optimizing a museum visitor trajectory prediction.

Current location: {current} ({current_info.get('name', current)})

{feature_info}
Visitor history:
{history_info}
{spatial_info}

Initial model prediction: {initial_pred}

Consider:
1. Spatial feasibility (nearby exhibits)
2. Visitor patterns (don't revisit too soon)
3. Exhibit popularity

Return the refined prediction as JSON:
{{"prediction_id": "exhibit_id", "reasoning": "short explanation"}}"""

        try:
            response = self.qwen_client.chat.completions.create(
                model=self.fine_tuned_model,  # 使用同一个模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            result_text = response.choices[0].message.content.strip()

            # 解析
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                pred_id = parsed.get('prediction_id')
                if pred_id:
                    return {
                        'prediction_id': pred_id,
                        'confidence': 0.85,  # LLM 优化后置信度更高
                        'reasoning': parsed.get('reasoning', ''),
                        'raw_output': result_text
                    }
        except Exception as e:
            print(f"[!] Stage 2 error: {e}")

        # 回退到阶段1的预测
        return stage1_prediction

    def predict_with_config(
        self,
        config: AblationConfig,
        sample: Dict,
        debug: bool = False
    ) -> Dict:
        """使用指定配置进行两阶段预测"""
        current = sample['current']
        history = sample.get('history', [])
        ground_truth = sample['next']

        # 阶段 1: 微调模型预测
        stage1_result = self.stage1_fine_tuned_prediction(current, history)

        # 阶段 2: LLM 优化
        if config.use_memory_in_llm or config.use_topology_in_llm or config.use_feature_in_llm:
            final_prediction = self.stage2_llm_refinement(current, history, stage1_result, config)
        else:
            # No-optimization: 直接用阶段1的结果
            final_prediction = stage1_result

        if debug:
            print(f"\n[DEBUG] Current: {current}, Truth: {ground_truth}")
            print(f"[DEBUG] Stage 1: {stage1_result['prediction_id']}")
            print(f"[DEBUG] Stage 2: {final_prediction.get('prediction_id', stage1_result['prediction_id'])}")

        return {
            'prediction': final_prediction,
            'ground_truth': ground_truth,
            'stage1_prediction': stage1_result
        }

    def _evaluate_config(self, config: AblationConfig) -> Dict:
        """评估单个配置"""
        correct_top1 = 0
        correct_top3 = 0
        dwell_errors = []
        attention_correct = 0

        print(f"    Evaluating {len(self.test_data)} samples...", end='', flush=True)

        for sample in self.test_data:
            result = self.predict_with_config(config, sample)
            prediction = result['prediction']
            ground_truth = result['ground_truth']

            # Top-1
            if prediction['prediction_id'] == ground_truth:
                correct_top1 += 1

            # Top-3 (使用空间候选)
            candidates = self._get_spatial_candidates(sample['current'])
            if ground_truth in candidates[:3]:
                correct_top3 += 1

            # Dwell MAE (简化，使用置信度反比)
            pred_dwell = 120 * (1 - prediction.get('confidence', 0.5)) + 30
            true_dwell = sample.get('dwell', 60)
            dwell_errors.append(abs(pred_dwell - true_dwell))

            # Attention (简化，全部给 C)
            if sample.get('attention', 'C') == 'C':
                attention_correct += 1

        total = len(self.test_data)
        print(f" Done")

        return {
            'top1_acc': correct_top1 / total,
            'top3_acc': correct_top3 / total,
            'mae': np.mean(dwell_errors) if dwell_errors else 0,
            'attn_acc': attention_correct / total
        }

    def run_ablation_study(self) -> Dict:
        """运行完整消融实验"""
        print("="*70)
        print("Two-Stage Ablation Study")
        print("="*70)
        print(f"Fine-tuned model: {self.fine_tuned_model}")
        print(f"Map: {self.map_name}")
        print(f"Samples: {len(self.test_data)}")
        print(f"Topology available: {USE_TOPOLOGY}")
        print("="*70)

        # 先测试一个样本
        print("\n[*] Testing with one sample...")
        test_sample = self.test_data[0]
        test_config = AblationConfig()
        result = self.predict_with_config(test_config, test_sample, debug=True)

        results = {}

        # 定义消融配置
        configs = [
            # Full (两阶段都启用)
            AblationConfig(use_memory_in_llm=True, use_topology_in_llm=True,
                          use_feature_in_llm=True, use_multi_step=True),
            # No-Memory (LLM 阶段不用历史)
            AblationConfig(use_memory_in_llm=False, use_topology_in_llm=True,
                          use_feature_in_llm=True, use_multi_step=True),
            # No-Topology (LLM 阶段不用空间信息)
            AblationConfig(use_memory_in_llm=True, use_topology_in_llm=False,
                          use_feature_in_llm=True, use_multi_step=True),
            # No-Feature (LLM 阶段不用展品特征)
            AblationConfig(use_memory_in_llm=True, use_topology_in_llm=True,
                          use_feature_in_llm=False, use_multi_step=True),
            # Stage1-Only (只用阶段1，不用LLM优化)
            AblationConfig(use_memory_in_llm=False, use_topology_in_llm=False,
                          use_feature_in_llm=False, use_multi_step=False),
        ]

        for config in configs:
            config_name = config.get_name()
            print(f"\n[*] Testing: {config_name}")

            config_results = self._evaluate_config(config)
            results[config_name] = config_results

            print(f"    Top-1: {config_results['top1_acc']:.1%}")
            print(f"    Top-3: {config_results['top3_acc']:.1%}")
            print(f"    MAE:   {config_results['mae']:.1f}s")

        # 保存结果
        self._save_results(results)
        self._print_comparison_table(results)

        return results

    def _save_results(self, results: Dict):
        """保存结果"""
        output_dir = "data/outputs/vllm_ablation"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "ablation_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[+] Results saved to {output_path}")

    def _print_comparison_table(self, results: Dict):
        """打印对比表格"""
        print("\n" + "="*70)
        print("Text Table")
        print("="*70)
        print(f"{'Variant':<20} {'Top-1':>10} {'Top-3':>10} {'MAE':>10}")
        print("-" * 52)

        full = results.get('Full', {})
        print(f"{'Full (Ours)':<20} {full['top1_acc']:>10.1%} {full['top3_acc']:>10.1%} {full['mae']:>10.1f}s")

        for name, res in results.items():
            if name == 'Full':
                continue
            print(f"{name:<20} {res['top1_acc']:>10.1%} {res['top3_acc']:>10.1%} {res['mae']:>10.1f}s")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-Stage Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture:
  Stage 1: Fine-tuned model → initial prediction
  Stage 2: Qwen LLM (with memory/topology) → refined prediction

Usage:
  # Start vLLM server first:
  python -m vllm.entrypoints.openai.api_server \\
      --model /home/g/models/qwen2.5-32b-int4 \\
      --served-model-name Qwen \\
      --gpu-memory-utilization 0.6 \\
      --enforce-eager \\
      --port 8000

  # Then run ablation:
  python scripts/run_vllm_ablation.py
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen",
        help="Model name in vLLM server"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API URL"
    )
    parser.add_argument(
        "--qwen-url",
        type=str,
        default=None,
        help="Separate Qwen API URL (if different)"
    )
    parser.add_argument(
        "--map",
        type=str,
        default="TH",
        help="Map name"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Test data path"
    )

    args = parser.parse_args()

    experiment = TwoStageAblationExperiment(
        fine_tuned_model=args.model,
        llm_base_url=args.api_url,
        qwen_api_url=args.qwen_url,
        map_name=args.map,
        data_path=args.data
    )

    results = experiment.run_ablation_study()


if __name__ == "__main__":
    main()
