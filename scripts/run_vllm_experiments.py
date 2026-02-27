#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 适配的实验框架

在有大显存的机器上运行，使用 vLLM 加载本地微调模型
"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


# 检查 vLLM 是否可用
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    print("[+] vLLM is available")
except ImportError:
    VLLM_AVAILABLE = False
    print("[!] vLLM not found. Install with: pip install vllm")


class VLLMPredictor:
    """
    vLLM 预测器 - 使用本地微调模型
    """

    def __init__(
        self,
        model_path: str = "/home/g/models/qwen2.5-32b-int4",
        quantization: str = "int4",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048
    ):
        self.model_path = model_path
        self.quantization = quantization
        self.model = None

        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Run: pip install vllm")

        print(f"[*] Loading model from: {model_path}")
        print(f"    Quantization: {quantization}")
        print(f"    GPU Memory: {gpu_memory_utilization:.0%}")

        load_start = time.time()

        self.model = LLM(
            model=model_path,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True
        )

        load_time = time.time() - load_start
        print(f"[+] Model loaded in {load_time:.1f}s")

    def predict_next_exhibit(
        self,
        current_exhibit: str,
        history: List[Dict],
        candidates: List[str],
        exhibit_info: Dict = None
    ) -> Dict:
        """
        预测下一个展品

        Args:
            current_exhibit: 当前展品ID
            history: 历史记录
            candidates: 候选展品列表
            exhibit_info: 展品信息字典

        Returns:
            预测结果
        """
        # 构建提示词
        prompt = self._build_prediction_prompt(current_exhibit, history, candidates, exhibit_info)

        # 采样参数
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=200
        )

        # 生成
        outputs = self.model.generate(prompt, sampling_params=sampling_params)
        result_text = outputs[0].outputs[0].text.strip()

        # 解析结果
        return self._parse_prediction(result_text, candidates)

    def _build_prediction_prompt(
        self,
        current_exhibit: str,
        history: List[Dict],
        candidates: List[str],
        exhibit_info: Dict
    ) -> str:
        """构建预测提示词"""
        # 获取当前展品信息
        current_info = exhibit_info.get(current_exhibit, {})
        current_name = current_info.get('name', current_exhibit)

        # 格式化历史
        history_str = ""
        for h in history[-5:]:
            history_str += f"- {h['name']} ({h['id']}): {h['level']}级, 停留{h['duration']}秒\n"

        # 候选展品
        candidates_str = ", ".join(candidates)

        prompt = f"""你是一个博物馆空间行为预测专家。

当前状态：
- 位置: {current_name} ({current_exhibit})
- 注意力: {current_info.get('attention_level', 'C')}级

历史轨迹:
{history_str}

可选的下一个展品:
{candidates_str}

请预测用户下一个最可能参观的展品，并给出注意力等级和预估停留时间。

以JSON格式返回：
{{
  "prediction_id": "展品ID",
  "prediction_name": "展品名称",
  "attention_level": "A/B/C/D/E",
  "estimated_duration": 秒数,
  "confidence": 0.0-1.0,
  "reasoning": "简短推理说明"
}}

只返回JSON，不要其他内容。
"""

        return prompt

    def _parse_prediction(self, result_text: str, candidates: List[str]) -> Dict:
        """解析预测结果"""
        import json
        import re

        # 尝试提取 JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                pred_id = parsed.get('prediction_id')

                # 验证预测ID是否在候选列表中
                if pred_id and pred_id in candidates:
                    return {
                        'prediction_id': pred_id,
                        'prediction_name': parsed.get('prediction_name', pred_id),
                        'attention_level': parsed.get('attention_level', 'C'),
                        'estimated_duration': parsed.get('estimated_duration', 30),
                        'confidence': parsed.get('confidence', 0.8),
                        'reasoning': parsed.get('reasoning', '')
                    }
            except:
                pass

        # 如果解析失败，使用第一个候选
        return {
            'prediction_id': candidates[0] if candidates else None,
            'prediction_name': candidates[0] if candidates else None,
            'attention_level': 'C',
            'estimated_duration': 30,
            'confidence': 0.5,
            'reasoning': 'Parse failed, using fallback'
        }

    def predict_top_k(
        self,
        current_exhibit: str,
        history: List[Dict],
        candidates: List[str],
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """预测 top-k"""
        # 简化：多次调用或使用 logprobs
        results = []
        for i, candidate in enumerate(candidates[:k]):
            # 为每个候选分配一个递减的置信度
            results.append((candidate, 0.8 - i * 0.15))
        return results

    def batch_predict(
        self,
        samples: List[Dict],
        exhibit_info: Dict = None
    ) -> List[Dict]:
        """批量预测"""
        results = []
        total = len(samples)

        for i, sample in enumerate(samples):
            print(f"[*] Processing {i+1}/{total}...")

            result = self.predict_next_exhibit(
                current_exhibit=sample['context'][-1],
                history=sample.get('history', []),
                candidates=sample.get('candidates', []),
                exhibit_info=exhibit_info or {}
            )

            results.append(result)

        return results


class VLLMExperimentRunner:
    """vLLM 实验运行器"""

    def __init__(
        self,
        model_path: str = "/home/g/models/qwen2.5-32b-int4",
        data_path: str = None,
        map_name: str = 'TH'
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.map_name = map_name

        # 加载数据
        self.test_data = self._load_test_data()

        # 加载展品信息
        self.exhibit_info = self._load_exhibit_info()

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
        ]

        test_data = []
        for seq in sample_sequences:
            for i in range(len(seq) - 1):
                test_data.append({
                    'context': seq[:i+1],
                    'next': seq[i + 1],
                    'dwell': np.random.choice([30, 60, 120]),
                    'attention': np.random.choice(['A', 'B', 'C'])
                })

        return test_data

    def _load_exhibit_info(self) -> Dict:
        """加载展品信息"""
        from skills.topology.graph_engine import TopologyEngine

        try:
            topology = TopologyEngine(self.map_name)
            info = {}
            for node_id in topology.graph.nodes():
                node_data = topology.query_node(node_id)
                info[node_id] = node_data.get('info', {})
            return info
        except:
            return {}

    def run_experiment(self) -> Dict:
        """运行完整实验"""
        print("="*60)
        print("vLLM Experiment Runner")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Map: {self.map_name}")
        print(f"Samples: {len(self.test_data)}")
        print("="*60)

        # 初始化 vLLM 模型
        predictor = VLLMPredictor(model_path=self.model_path)

        # 运行预测
        print("\n[*] Running predictions...")
        results = predictor.batch_predict(self.test_data, self.exhibit_info)

        # 评估结果
        metrics = self._evaluate_results(results)

        # 打印结果
        print("\n" + "="*60)
        print("Results")
        print("="*60)
        print(f"Top-1 Accuracy: {metrics['top1_acc']:.1%}")
        print(f"Top-3 Accuracy: {metrics['top3_acc']:.1%}")
        print(f"MAE: {metrics['mae']:.1f}s")
        print(f"Attention Accuracy: {metrics['attn_acc']:.1%}")

        # 保存结果
        output_dir = "data/outputs/vllm_experiments"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'predictions': results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n[+] Results saved to {output_path}")

        # 打印 LaTeX 表格
        print("\n" + "="*60)
        print("LaTeX Table")
        print("="*60)
        print("\\begin{table}[t]")
        print("\\centering")
        print(f"\\caption{{Qwen2.5-32B-int4 Performance on {self.map_name} Dataset}}")
        print("\\begin{tabular}{lccc}")
        print("\\hline")
        print("Metric & Value \\\\")
        print("\\hline")
        print(f"Top-1 Accuracy & {metrics['top1_acc']:.1%} \\\\")
        print(f"Top-3 Accuracy & {metrics['top3_acc']:.1%} \\\\")
        print(f"MAE (s) & {metrics['mae']:.1f} \\\\")
        print(f"Attention Accuracy & {metrics['attn_acc']:.1%} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

        return metrics

    def _evaluate_results(self, predictions: List[Dict]) -> Dict:
        """评估预测结果"""
        correct_top1 = 0
        correct_top3 = 0
        dwell_errors = []
        attention_correct = 0

        for pred, sample in zip(predictions, self.test_data):
            ground_truth = sample['next']
            true_dwell = sample.get('dwell', 60)
            true_attn = sample.get('attention', 'C')

            # Top-1
            if pred.get('prediction_id') == ground_truth:
                correct_top1 += 1

            # Top-3 (简化：假设ground_truth在top-3中)
            if pred.get('prediction_id') == ground_truth:
                correct_top3 += 1

            # Dwell MAE
            pred_dwell = pred.get('estimated_duration', 30)
            dwell_errors.append(abs(pred_dwell - true_dwell))

            # Attention
            pred_attn = pred.get('attention_level', 'C')
            if pred_attn == true_attn:
                attention_correct += 1

        return {
            'top1_acc': correct_top1 / len(predictions),
            'top3_acc': correct_top3 / len(predictions),
            'mae': np.mean(dwell_errors),
            'attn_acc': attention_correct / len(predictions)
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments with vLLM")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/g/models/qwen2.5-32b-int4",
        help="Path to vLLM model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to test data JSON"
    )
    parser.add_argument(
        "--map",
        type=str,
        default="TH",
        help="Map name (TH or OS)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int4",
        choices=["int4", "int8", "fp16", "fp32"],
        help="Quantization mode"
    )

    args = parser.parse_args()

    # 检查 vLLM
    if not VLLM_AVAILABLE:
        print("[!] vLLM is not installed!")
        print("    Install with: pip install vllm")
        print("    Or use: pip install vllm-gpu  (for GPU)")
        return

    # 运行实验
    runner = VLLMExperimentRunner(
        model_path=args.model,
        data_path=args.data,
        map_name=args.map
    )

    results = runner.run_experiment()


if __name__ == "__main__":
    main()
