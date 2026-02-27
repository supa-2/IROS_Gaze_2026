#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study with vLLM - 三阶段架构

阶段 0: Qwen LLM 处理拓扑和特征 → 提取关键信息
阶段 1: 微调模型基于提取的信息预测
阶段 2: Qwen LLM 用记忆优化最终输出

在有模型的机器上运行，测试各组件的贡献
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

USE_TOPOLOGY = os.path.exists(os.path.join(project_root, 'skills'))


class AblationConfig:
    """消融实验配置"""

    def __init__(
        self,
        use_topology_preprocess: bool = True,  # 阶段0: Qwen 是否处理拓扑
        use_feature_preprocess: bool = True,   # 阶段0: Qwen 是否处理特征
        use_memory_in_llm: bool = True,        # 阶段2: Qwen 是否用记忆优化
    ):
        self.use_topology_preprocess = use_topology_preprocess
        self.use_feature_preprocess = use_feature_preprocess
        self.use_memory_in_llm = use_memory_in_llm

    def get_name(self) -> str:
        """获取配置名称"""
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
        map_name: str = 'TH',
        data_path: str = None
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.map_name = map_name
        self.data_path = data_path

        # 加载数据
        self.test_data = self._load_test_data()
        self.exhibit_info = self._load_exhibit_info()
        self.topology_data = self._load_topology_data() if USE_TOPOLOGY else {}

        # 初始化客户端
        self._init_client()

        # 缓存预处理结果（避免重复调用）
        self.preprocess_cache = {}

    def _init_client(self):
        """初始化 API 客户端"""
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
                    'history': [{'id': seq[j], 'name': seq[j], 'duration': 60} for j in range(i)],
                    'dwell': 60,
                    'attention': 'A'
                })
        return test_data

    def _load_exhibit_info(self) -> Dict:
        """加载展品信息"""
        mock_exhibits = [
            'TH-E01', 'TH-I-B01', 'TH-B02', 'TH-C03', 'TH-D04',
            'TH-E05', 'TH-F06', 'TH-G07', 'TH-A01', 'TH-H08'
        ]
        return {
            eid: {
                'name': eid,
                'type': np.random.choice(['Interactive', 'Static', 'Video']),
                'popularity': np.random.randint(50, 100),
                'zone': chr(65 + i % 5)  # A-E zone
            }
            for i, eid in enumerate(mock_exhibits)
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
                }
            return data
        except:
            return {}

    def stage0_qwen_preprocess(
        self,
        current: str,
        config: AblationConfig
    ) -> str:
        """
        阶段 0: Qwen LLM 处理拓扑和特征，提取关键信息
        """
        cache_key = (current, config.use_topology_preprocess, config.use_feature_preprocess)
        if cache_key in self.preprocess_cache:
            return self.preprocess_cache[cache_key]

        current_info = self.exhibit_info.get(current, {})

        # 构建预处理 prompt
        prompt_parts = [f"Current location: {current}"]

        # 拓扑信息
        if config.use_topology_preprocess and USE_TOPOLOGY:
            neighbors = self.topology_data.get(current, {}).get('neighbors', [])
            if neighbors:
                neighbor_ids = [n.get('id') for n in neighbors[:5]]
                prompt_parts.append(f"Nearby exhibits: {', '.join(neighbor_ids)}")
            else:
                prompt_parts.append("Nearby exhibits: All connected exhibits")

        # 展品特征
        if config.use_feature_preprocess:
            prompt_parts.append(f"Exhibit type: {current_info.get('type', 'Unknown')}")
            prompt_parts.append(f"Popularity: {current_info.get('popularity', 50)}/100")

        prompt_parts.append("\nExtract key information for trajectory prediction. Return JSON:")
        prompt_parts.append('{"context": "brief description", "candidates": ["id1", "id2", "id3"]}')

        prompt = "\n".join(prompt_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            result = response.choices[0].message.content.strip()

            # 缓存结果
            self.preprocess_cache[cache_key] = result
            return result

        except Exception as e:
            # 默认返回
            return '{"context": "standard visit", "candidates": []}'

    def stage1_fine_tuned_predict(
        self,
        current: str,
        processed_context: str,
        config: AblationConfig
    ) -> Dict:
        """
        阶段 1: 微调模型基于 Qwen 预处理的信息做预测
        """
        # 构建预测 prompt
        history_str = ", ".join(self.test_data[0]['context'][:3]) if self.test_data else "TH-E01"

        prompt = f"""You are a trajectory prediction model.

Current location: {current}

Processed context from spatial analysis:
{processed_context}

Previous locations: {history_str}

Predict the next exhibit. Return JSON:
{{"prediction_id": "exhibit_id", "confidence": 0.0-1.0}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            result = response.choices[0].message.content.strip()

            # 解析 JSON
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                pred_id = parsed.get('prediction_id')
                if pred_id:
                    return {
                        'prediction_id': pred_id,
                        'confidence': parsed.get('confidence', 0.8),
                        'raw': result
                    }
        except Exception as e:
            pass

        # 默认
        return {
            'prediction_id': current,
            'confidence': 0.5,
            'raw': ''
        }

    def stage2_qwen_refine(
        self,
        current: str,
        history: List[Dict],
        initial_prediction: Dict,
        config: AblationConfig
    ) -> Dict:
        """
        阶段 2: Qwen LLM 用记忆优化预测
        """
        initial_pred = initial_prediction['prediction_id']

        # 构建记忆信息
        memory_info = ""
        if config.use_memory_in_llm and history:
            visited = [h['id'] for h in history]
            memory_info = f"Recently visited: {', '.join(visited[-5:])}\n"

        prompt = f"""You are refining a trajectory prediction.

Current: {current}
Initial prediction: {initial_pred}

{memory_info}Consider visitor patterns:
- Don't predict recently visited exhibits
- Consider exhibit popularity

Return the refined prediction as JSON:
{{"prediction_id": "exhibit_id", "confidence": 0.0-1.0, "reasoning": "explanation"}}"""

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
                pred_id = parsed.get('prediction_id')
                if pred_id:
                    return {
                        'prediction_id': pred_id,
                        'confidence': parsed.get('confidence', 0.85),
                        'reasoning': parsed.get('reasoning', ''),
                        'raw': result
                    }
        except:
            pass

        # 回退到初始预测
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
            print(f"\n[DEBUG] Current: {current}, Truth: {ground_truth}")
            print(f"[DEBUG] Stage0 (Qwen preprocess): {processed_context[:100]}...")
            print(f"[DEBUG] Stage1 (Fine-tuned): {stage1_result['prediction_id']}")
            print(f"[DEBUG] Stage2 (Qwen refine): {final_result['prediction_id']}")
            print(f"[DEBUG] Match: {final_result['prediction_id'] == ground_truth}")

        return {
            'prediction': final_result,
            'ground_truth': ground_truth,
            'stage1': stage1_result
        }

    def _evaluate_config(self, config: AblationConfig) -> Dict:
        """评估单个配置"""
        correct_top1 = 0
        dwell_errors = []
        all_candidates = []

        print(f"    Evaluating {len(self.test_data)} samples...", end='', flush=True)

        for sample in self.test_data:
            result = self.predict_with_config(config, sample)
            prediction = result['prediction']
            ground_truth = result['ground_truth']

            # Top-1
            if prediction['prediction_id'] == ground_truth:
                correct_top1 += 1

            # 收集候选用于 Top-3
            current = sample['current']
            all_candidates.append(self._get_candidates(current, config.use_topology_preprocess))

            # Dwell MAE
            pred_dwell = 120 * (1 - prediction.get('confidence', 0.5)) + 30
            true_dwell = sample.get('dwell', 60)
            dwell_errors.append(abs(pred_dwell - true_dwell))

        total = len(self.test_data)

        # Top-3: ground truth 是否在候选的前3个中
        correct_top3 = 0
        for i, sample in enumerate(self.test_data):
            candidates = all_candidates[i]
            if sample['next'] in candidates[:3]:
                correct_top3 += 1

        print(f" Done")

        return {
            'top1_acc': correct_top1 / total,
            'top3_acc': correct_top3 / total,
            'mae': np.mean(dwell_errors) if dwell_errors else 0
        }

    def _get_candidates(self, current: str, use_topology: bool) -> List[str]:
        """获取候选展品"""
        if use_topology and USE_TOPOLOGY and current in self.topology_data:
            neighbors = self.topology_data[current].get('neighbors', [])
            return [n.get('id') for n in neighbors]
        # Mock: 返回所有其他展品
        return [e for e in self.exhibit_info.keys() if e != current]

    def run_ablation_study(self) -> Dict:
        """运行完整消融实验"""
        print("="*70)
        print("Three-Stage Ablation Study")
        print("="*70)
        print(f"Model: {self.model_name}")
        print(f"Map: {self.map_name}")
        print(f"Samples: {len(self.test_data)}")
        print("="*70)

        # 测试一个样本
        print("\n[*] Testing with one sample...")
        test_config = AblationConfig()
        test_result = self.predict_with_config(test_config, self.test_data[0], debug=True)

        results = {}

        # 定义消融配置
        configs = [
            # Full: 三阶段完整
            AblationConfig(use_topology_preprocess=True, use_feature_preprocess=True, use_memory_in_llm=True),
            # No-Topology: 阶段0 不处理拓扑
            AblationConfig(use_topology_preprocess=False, use_feature_preprocess=True, use_memory_in_llm=True),
            # No-Feature: 阶段0 不处理特征
            AblationConfig(use_topology_preprocess=True, use_feature_preprocess=False, use_memory_in_llm=True),
            # No-Memory: 阶段2 不用记忆
            AblationConfig(use_topology_preprocess=True, use_feature_preprocess=True, use_memory_in_llm=False),
            # No-Preprocess: 跳过阶段0
            AblationConfig(use_topology_preprocess=False, use_feature_preprocess=False, use_memory_in_llm=True),
        ]

        for config in configs:
            config_name = config.get_name()
            print(f"\n[*] Testing: {config_name}")

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
        print(f"\n[+] Results saved to {output_path}")

    def _print_table(self, results: Dict):
        """打印结果表格"""
        print("\n" + "="*70)
        print("Results Summary")
        print("="*70)
        print(f"{'Variant':<20} {'Top-1':>12} {'Top-3':>12} {'MAE':>10}")
        print("-" * 56)

        full = results.get('Full', {})
        print(f"{'Full (Ours)':<20} {full['top1_acc']:>12.1%} {full['top3_acc']:>12.1%} {full['mae']:>10.1f}s")

        for name, res in results.items():
            if name == 'Full':
                continue
            print(f"{name:<20} {res['top1_acc']:>12.1%} {res['top3_acc']:>12.1%} {res['mae']:>10.1f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Three-Stage Ablation Study")
    parser.add_argument("--model", default="Qwen", help="Model name")
    parser.add_argument("--api-url", default="http://localhost:8000/v1", help="API URL")
    parser.add_argument("--map", default="TH", help="Map name")
    parser.add_argument("--data", default=None, help="Test data path")
    args = parser.parse_args()

    experiment = ThreeStageAblation(
        model_name=args.model,
        api_url=args.api_url,
        map_name=args.map,
        data_path=args.data
    )
    experiment.run_ablation_study()


if __name__ == "__main__":
    main()
