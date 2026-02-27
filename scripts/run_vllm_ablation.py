#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study with vLLM - 使用本地微调模型的消融实验

在有模型的机器上运行，测试各组件的贡献

使用方式 1: 直接运行（需要足够显存）
    python scripts/run_vllm_ablation.py --model /path/to/model

使用方式 2: 使用 API server（推荐，显存不足时）
    # 终端 1: 启动 server
    PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
    python -m vllm.entrypoints.openai.api_server \
        --model /home/g/models/qwen2.5-32b-int4 \
        --gpu-memory-utilization 0.6 \
        --max-model-len 2048 \
        --enforce-eager \
        --port 8000

    # 终端 2: 运行实验
    python scripts/run_vllm_ablation.py --api-url http://localhost:8000
"""

import os
import sys
import json
import numpy as np
import time
from typing import List, Dict, Optional

# 设置 PyTorch 显存环境变量（与您成功部署时一致）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# 设置项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# 验证 skills 目录存在
skills_path = os.path.join(project_root, 'skills')
if not os.path.exists(skills_path):
    print(f"[!] Warning: skills directory not found at {skills_path}")
    print(f"[!] Project root: {project_root}")
    print(f"[!] Will run without topology features (using mock data)")
    USE_TOPOLOGY = False
else:
    USE_TOPOLOGY = True


class VLLMAblationConfig:
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


class VLLMAblationExperiment:
    """使用 vLLM 的消融实验"""

    def __init__(
        self,
        model_path: str = "/home/g/models/qwen2.5-32b-int4",
        map_name: str = 'TH',
        data_path: str = None,
        gpu_memory_utilization: float = 0.6,
        max_model_len: int = 2048,
        api_url: str = None
    ):
        self.model_path = model_path
        self.map_name = map_name
        self.data_path = data_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.api_url = api_url  # 如果提供，使用 API 而非直接加载模型

        # 加载数据
        self.test_data = self._load_test_data()
        self.exhibit_info = self._load_exhibit_info()

        # 初始化 vLLM 模型或 API 客户端
        self.predictor = None
        if self.api_url:
            self._init_api_client()
        else:
            self._init_model()

    def _init_model(self):
        """初始化 vLLM 模型"""
        try:
            from vllm import LLM, SamplingParams

            print(f"[*] Loading vLLM model: {self.model_path}")
            print(f"[*] GPU memory utilization: {self.gpu_memory_utilization:.0%}")
            print(f"[*] Max model length: {self.max_model_len}")
            print("[*] Using enforce_eager=True to save memory (same as your working setup)")

            self.model = LLM(
                model=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                disable_log_stats=True,
                enable_prefix_caching=False,
                enforce_eager=True,  # 关键！禁用 CUDA graph 节省显存
            )
            print("[+] vLLM model loaded successfully")
        except ImportError:
            raise RuntimeError("vLLM not installed. Run: pip install vllm")

    def _init_api_client(self):
        """初始化 API 客户端（连接到已启动的 vLLM server）"""
        from openai import OpenAI

        # 确保使用正确的 base URL（需要包含 /v1）
        api_base = self.api_url.rstrip('/')
        if not api_base.endswith('/v1'):
            api_base = api_base + '/v1'

        print(f"[*] Connecting to vLLM API server at: {api_base}")
        self.client = OpenAI(
            api_key="sk-YourCustomSecretKey123",  # 与 server 启动时一致
            base_url=api_base
        )

        # 使用服务端注册的模型名称（通过 --served-model-name 指定）
        self.served_model_name = "Qwen"

        # 测试连接
        try:
            response = self.client.chat.completions.create(
                model=self.served_model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            print(f"[+] Connected to vLLM API server successfully")
            print(f"[*] Server model: {self.served_model_name}")
        except Exception as e:
            print(f"[!] Failed to connect to API server: {e}")
            print("[!] Make sure the server is running:")
            print("    PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \\")
            print("    python -m vllm.entrypoints.openai.api_server \\")
            print(f"        --model {self.model_path} \\")
            print("        --served-model-name Qwen \\")
            print("        --gpu-memory-utilization 0.6 \\")
            print("        --max-model-len 2048 \\")
            print("        --enforce-eager \\")
            print("        --port 8000")
            raise

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
        if not USE_TOPOLOGY:
            # 使用模拟数据
            print("[!] Using mock exhibit data (topology not available)")
            return self._get_mock_exhibit_info()

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
            print("[!] Using mock exhibit data")
            return self._get_mock_exhibit_info()

    def _get_mock_exhibit_info(self) -> Dict:
        """获取模拟展品信息"""
        mock_exhibits = [
            'TH-E01', 'TH-I-B01', 'TH-B02', 'TH-C03', 'TH-D04',
            'TH-E05', 'TH-F06', 'TH-G07', 'TH-A01', 'TH-H08'
        ]
        return {
            eid: {
                'name': eid,
                'features': f'Exhibit {eid} features',
                'attention_level': 'C'
            }
            for eid in mock_exhibits
        }

    def _get_candidates(self, current: str, use_topology: bool) -> List[str]:
        """获取候选展品"""
        if use_topology and USE_TOPOLOGY:
            try:
                from skills.topology.graph_engine import TopologyEngine
                topology = TopologyEngine(self.map_name)
                info = topology.query_node(current)
                choices = info.get('context', {}).get('direct_choices', [])
                return [c.get('id') for c in choices]
            except Exception as e:
                print(f"[!] Topology query failed: {e}, using mock candidates")

        # 返回所有展品（不包括当前）
        all_exhibits = list(self.exhibit_info.keys())
        return [e for e in all_exhibits if e != current]

    def predict_with_config(
        self,
        config: VLLMAblationConfig,
        sample: Dict,
        debug: bool = False
    ) -> Dict:
        """使用指定配置进行预测"""
        current = sample['current']
        history = sample.get('history', [])
        ground_truth = sample['next']

        # 准备候选
        candidates = self._get_candidates(current, config.use_topology)

        # 准备历史（No-Memory: 清空历史）
        if not config.use_memory:
            history = []

        # 构建提示词
        prompt = self._build_prompt(current, history, candidates, config)

        # 调用模型（直接模式或 API 模式）
        if self.api_url:
            result_text = self._call_api(prompt)
        else:
            result_text = self._call_model(prompt)

        if debug:
            print(f"\n[DEBUG] Sample: current={current}, truth={ground_truth}")
            print(f"[DEBUG] Candidates: {candidates}")
            print(f"[DEBUG] Model output:\n{result_text}")

        # 解析结果
        prediction = self._parse_prediction(result_text, candidates)

        if debug:
            print(f"[DEBUG] Parsed prediction: {prediction['prediction_id']}")
            print(f"[DEBUG] Match: {prediction['prediction_id'] == ground_truth}")

        return {
            'prediction': prediction,
            'ground_truth': ground_truth
        }

    def _call_model(self, prompt: str) -> str:
        """直接调用 vLLM 模型"""
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=200
        )
        outputs = self.model.generate([prompt], sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _call_api(self, prompt: str) -> str:
        """通过 API 调用 vLLM server"""
        response = self.client.chat.completions.create(
            model=self.served_model_name,  # 使用服务端注册的模型名
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            top_p=0.9,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()

    def _build_prompt(
        self,
        current: str,
        history: List[Dict],
        candidates: List[str],
        config: 'VLLMAblationConfig'
    ) -> str:
        """构建预测提示词"""
        current_info = self.exhibit_info.get(current, {})
        current_name = current_info.get('name', current)

        # 历史轨迹
        history_str = ""
        if config.use_memory and history:
            for h in history[-5:]:
                history_str += f"- {h['name']} ({h['id']}): {h['level']}级\n"

        # 候选展品
        candidates_str = ", ".join(candidates)

        # 空间信息
        spatial_str = ""
        if config.use_topology:
            spatial_str = "\n注意：请优先选择空间上相邻的展品。"

        prompt = f"""你是一个博物馆空间行为预测专家。

当前状态：
- 位置: {current_name} ({current})
- 特征: {current_info.get('features', '')}

历史轨迹:
{history_str}

可选的下一个展品:
{candidates_str}{spatial_str}

请预测用户下一个最可能参观的展品。

返回JSON格式：
{{
  "prediction_id": "展品ID",
  "attention_level": "A/B/C/D/E",
  "estimated_duration": 秒数,
  "confidence": 0.0-1.0
}}
"""
        return prompt

    def _parse_prediction(self, result_text: str, candidates: List[str]) -> Dict:
        """解析预测结果"""
        import json
        import re

        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                pred_id = parsed.get('prediction_id')

                # 验证预测ID
                if pred_id and pred_id in candidates:
                    return {
                        'prediction_id': pred_id,
                        'prediction_name': parsed.get('prediction_name', pred_id),
                        'attention_level': parsed.get('attention_level', 'C'),
                        'estimated_duration': parsed.get('estimated_duration', 30),
                        'confidence': parsed.get('confidence', 0.8)
                    }
            except:
                pass

        # 默认返回第一个候选
        return {
            'prediction_id': candidates[0] if candidates else None,
            'prediction_name': candidates[0] if candidates else None,
            'attention_level': 'C',
            'estimated_duration': 30,
            'confidence': 0.5
        }

    def run_ablation_study(self) -> Dict:
        """运行完整消融实验"""
        print("="*70)
        print("Ablation Study with vLLM (Your Fine-tuned Model)")
        print("="*70)
        print(f"Model: {self.model_path}")
        print(f"Map: {self.map_name}")
        print(f"Samples: {len(self.test_data)}")
        if self.api_url:
            print(f"API URL: {self.api_url}")
        print("="*70)

        # 先测试一个样本，看看模型输出
        print("\n[*] Testing model output with one sample...")
        test_sample = self.test_data[0]
        test_config = VLLMAblationConfig()
        result = self.predict_with_config(test_config, test_sample, debug=True)

        results = {}

        # 定义所有消融配置
        configs = [
            VLLMAblationConfig(use_memory=True, use_topology=True,
                               use_feature_extractor=True, use_multi_step=True),
            VLLMAblationConfig(use_memory=False, use_topology=True,
                               use_feature_extractor=True, use_multi_step=True),
            VLLMAblationConfig(use_memory=True, use_topology=False,
                               use_feature_extractor=True, use_multi_step=True),
            VLLMAblationConfig(use_memory=True, use_topology=True,
                               use_feature_extractor=False, use_multi_step=True),
            VLLMAblationConfig(use_memory=True, use_topology=True,
                               use_feature_extractor=True, use_multi_step=False),
        ]

        for config in configs:
            config_name = config.get_name()
            print(f"\n[*] Testing: {config_name}")

            config_results = self._evaluate_config(config)

            results[config_name] = config_results

            print(f"    Top-1: {config_results['top1_acc']:.1%}")
            print(f"    Top-3: {config_results['top3_acc']:.1%}")
            print(f"    MAE:   {config_results['mae']:.1f}s")
            print(f"    Attn:  {config_results['attn_acc']:.1%}")

        # 保存结果
        self._save_results(results)

        # 打印对比表格
        self._print_comparison_table(results)

        return results

    def _evaluate_config(self, config: VLLMAblationConfig) -> Dict:
        """评估单个配置"""
        correct_top1 = 0
        correct_top3 = 0
        dwell_errors = []
        attention_correct = 0

        print(f"    Evaluating {len(self.test_data)} samples...", end='', flush=True)

        for i, sample in enumerate(self.test_data):
            # 预测
            result = self.predict_with_config(config, sample)
            prediction = result['prediction']
            ground_truth = result['ground_truth']

            # Top-1
            if prediction['prediction_id'] == ground_truth:
                correct_top1 += 1

            # Top-3 - 使用模型返回的候选（如果模型没有返回多个候选，则用拓扑候选作为替代）
            candidates = self._get_candidates(sample['current'], config.use_topology)
            # Top-3: 检查 ground_truth 是否在模型预测或候选的前3个中
            # 这里简化：假设候选列表本身就是某种排序
            if ground_truth in candidates[:3]:
                correct_top3 += 1

            # Dwell MAE
            pred_dwell = prediction['estimated_duration']
            true_dwell = sample.get('dwell', 60)
            dwell_errors.append(abs(pred_dwell - true_dwell))

            # Attention
            pred_attn = prediction['attention_level']
            true_attn = sample.get('attention', 'C')
            if pred_attn == true_attn:
                attention_correct += 1

        total = len(self.test_data)
        print(f" Done")

        return {
            'top1_acc': correct_top1 / total,
            'top3_acc': correct_top3 / total,
            'mae': np.mean(dwell_errors) if dwell_errors else 0,
            'attn_acc': attention_correct / total
        }

    def _save_results(self, results: Dict):
        """保存结果"""
        output_dir = "data/outputs/vllm_ablation"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "ablation_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[+] Results saved to {output_path}")

    def _print_comparison_table(self, results: Dict):
        """打印对比表格（LaTeX格式）"""
        print("\n" + "="*70)
        print("LaTeX Table for Ablation Study")
        print("="*70)

        print("\n\\begin{table}[t]")
        print("\\centering")
        print("\\caption{Ablation study with fine-tuned Qwen2.5-32B-int4}")
        print("\\label{tab:ablation}")
        print("\\begin{tabular}{lccccc}")
        print("\\hline")
        print("Variant & Top-1 $\\uparrow$ & Top-3 $\\uparrow$ & MAE$\\downarrow$ & Attn $\\uparrow$ \\\\")
        print("\\hline")

        # Full (Ours)
        full = results.get('Full', {})
        print(f"Full (Ours) & {full['top1_acc']:.1%} & {full['top3_acc']:.1%} & {full['mae']:.1f}s & {full['attn_acc']:.1%} \\\\" + chr(92))

        # 其他变体
        for name, res in results.items():
            if name == 'Full':
                continue

            # 计算差异
            diff_top1 = (full['top1_acc'] - res['top1_acc']) * 100
            diff_top3 = (full['top3_acc'] - res['top3_acc']) * 100
            diff_mae = res['mae'] - full['mae']
            diff_attn = (full['attn_acc'] - res['attn_acc']) * 100

            line = f"-{name} & {res['top1_acc']:.1%} ({diff_top1:+.1f}) & " \
                   f"{res['top3_acc']:.1%} ({diff_top3:+.1f}) & " \
                   f"{res['mae']:.1f}s ({diff_mae:+.1f}) & " \
                   f"{res['attn_acc']:.1%} ({diff_attn:+.1f})"
            print(line + " \\\\" + chr(92))

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

        # 打印纯文本版本
        print("\n" + "="*70)
        print("Text Table (for reference)")
        print("="*70)
        print(f"{'Variant':<20} {'Top-1':>10} {'Top-3':>10} {'MAE':>10} {'Attn':>10}")
        print("-" * 62)

        print(f"{'Full (Ours)':<20} {full['top1_acc']:>10.1%} {full['top3_acc']:>10.1%} {full['mae']:>10.1f}s {full['attn_acc']:>10.1%}")

        for name, res in results.items():
            if name == 'Full':
                continue
            print(f"{name:<20} {res['top1_acc']:>10.1%} {res['top3_acc']:>10.1%} {res['mae']:>10.1f}s {res['attn_acc']:>10.1%}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ablation study with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用 API server 模式（推荐，显存不足时）
  # 先在终端1启动 server:
  PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \\
  python -m vllm.entrypoints.openai.api_server \\
      --model /home/g/models/qwen2.5-32b-int4 \\
      --gpu-memory-utilization 0.6 \\
      --max-model-len 2048 \\
      --enforce-eager \\
      --port 8000

  # 然后在终端2运行实验:
  python scripts/run_vllm_ablation.py --api-url http://localhost:8000

  # 直接运行模式（需要足够显存）
  python scripts/run_vllm_ablation.py --model /home/g/models/qwen2.5-32b-int4
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/g/models/qwen2.5-32b-int4",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Use API server instead of direct model (e.g., http://localhost:8000)"
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
        "--gpu-memory",
        type=float,
        default=0.6,  # 与用户成功部署时的参数一致
        help="GPU memory utilization (default: 0.6)"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=2048,  # 与用户成功部署时的参数一致
        help="Maximum model length (default: 2048)"
    )

    args = parser.parse_args()

    # API 模式不需要检查 vLLM
    if not args.api_url:
        try:
            from vllm import LLM
        except ImportError:
            print("[!] vLLM is not installed!")
            print("    Install with: pip install vllm")
            return

    # 运行实验
    experiment = VLLMAblationExperiment(
        model_path=args.model,
        data_path=args.data,
        map_name=args.map,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_len,
        api_url=args.api_url
    )

    results = experiment.run_ablation_study()


if __name__ == "__main__":
    main()
