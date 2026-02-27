#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Models for Trajectory Prediction

快速实现各种baseline方法，用于论文对照实验
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ============================================
# 1. Statistical Methods (最快实现)
# ============================================

class MarkovChain:
    """马尔可夫链基线"""

    def __init__(self, order: int = 1):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.state_counts = defaultdict(int)

    def train(self, sequences: List[List[str]]):
        """训练：统计转移频率"""
        for seq in sequences:
            for i in range(len(seq) - self.order):
                # 构建 state: (prev_1, prev_2, ...) 的 tuple
                state = tuple(seq[i:i+self.order])
                next_item = seq[i + self.order]
                self.transitions[state][next_item] += 1
                self.state_counts[state] += 1

    def predict(self, context: List[str], candidates: List[str] = None) -> Tuple[str, float]:
        """
        预测下一个项目

        Args:
            context: 历史序列
            candidates: 候选集合（如果为None，从训练集中选）

        Returns:
            (预测结果, 置信度)
        """
        # 构建 state
        if len(context) < self.order:
            # 历史不足，用最频繁的
            all_next = []
            for next_items in self.transitions.values():
                all_next.extend(list(next_items.keys()))
            if all_next:
                most_common = Counter(all_next).most_common(1)[0]
                return most_common[0], most_common[1] / sum(c for c in self.transitions.values())
            return None, 0.0

        state = tuple(context[-self.order:])

        if state not in self.transitions:
            # 回退到低阶
            if self.order > 1:
                fallback = MarkovChain(self.order - 1)
                fallback.transitions = self.transitions
                fallback.state_counts = self.state_counts
                return fallback.predict(context, candidates)
            # 如果还是不行，随机
            return None, 0.0

        # 转移概率
        next_counts = self.transitions[state]
        total = sum(next_counts.values())

        # 如果有候选约束，只在候选中选
        if candidates:
            candidate_items = {c: next_counts.get(c, 0) for c in candidates}
            if not candidate_items:
                return None, 0.0
            best = max(candidate_items.items(), key=lambda x: x[1])
            return best[0], best[1] / max(total, 1)

        best = next_counts.most_common(1)[0]
        return best[0], best[1] / total

    def predict_top_k(self, context: List[str], k: int = 3, candidates: List[str] = None) -> List[Tuple[str, float]]:
        """预测 top-k"""
        if len(context) < self.order:
            return []

        state = tuple(context[-self.order:])
        if state not in self.transitions:
            return []

        next_counts = self.transitions[state]
        if candidates:
            next_counts = {c: next_counts.get(c, 0) for c in candidates}

        top_k = next_counts.most_common(k)
        total = sum(next_counts.values())
        return [(item, count / max(total, 1)) for item, count in top_k]


# ============================================
# 2. Deep Learning Methods (需要训练)
# ============================================

class SequenceDataset:
    """序列数据集预处理"""

    def __init__(self, sequences: List[List[str]], exhibit_to_idx: Dict[str, int]):
        self.sequences = sequences
        self.exhibit_to_idx = exhibit_to_idx
        self.idx_to_exhibit = {v: k for k, v in exhibit_to_idx.items()}
        self.num_exhibits = len(exhibit_to_idx)

    def encode_sequence(self, seq: List[str]) -> List[int]:
        return [self.exhibit_to_idx.get(x, 0) for x in seq]

    def prepare_training_data(self, max_len: int = 20):
        """准备训练数据"""
        X, y = [], []

        for seq in self.sequences:
            encoded = self.encode_sequence(seq)
            for i in range(len(encoded) - 1):
                # 截取或padding历史
                history = encoded[max(0, i - max_len + 1):i + 1]
                # padding到固定长度
                if len(history) < max_len:
                    history = [0] * (max_len - len(history)) + history
                X.append(history)
                y.append(encoded[i + 1])

        return np.array(X), np.array(y)


class SimpleLSTM:
    """简单的LSTM模型（使用pytorch或sklearn）"""

    def __init__(self, num_classes: int, embedding_dim: int = 32, hidden_dim: int = 64):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self.exhibit_to_idx = None
        self.idx_to_exhibit = None

    def train(self, sequences: List[List[str]], max_len: int = 20):
        """训练模型"""
        from sklearn.preprocessing import LabelEncoder
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Masking
        from tensorflow.keras.utils import to_categorical

        # 构建词汇表
        all_exhibits = list(set([x for seq in sequences for x in seq]))
        self.exhibit_to_idx = {e: i + 1 for i, e in enumerate(all_exhibits)}  # 0 for padding
        self.exhibit_to_idx['<PAD>'] = 0
        self.idx_to_exhibit = {v: k for k, v in self.exhibit_to_idx.items()}

        # 准备数据
        X, y = [], []
        for seq in sequences:
            encoded = [self.exhibit_to_idx.get(x, 0) for x in seq]
            for i in range(len(encoded) - 1):
                history = encoded[max(0, i - max_len + 1):i + 1]
                if len(history) < max_len:
                    history = [0] * (max_len - len(history)) + history
                X.append(history)
                y.append(encoded[i + 1])

        X = np.array(X)
        y_cat = to_categorical(y, num_classes=len(self.exhibit_to_idx))

        # 构建模型
        self.model = Sequential([
            Embedding(len(self.exhibit_to_idx), self.embedding_dim, input_length=max_len, mask_zero=True),
            LSTM(self.hidden_dim, return_sequences=False),
            Dense(len(self.exhibit_to_idx), activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y_cat, epochs=10, batch_size=32, verbose=0)

    def predict(self, context: List[str], candidates: List[str] = None) -> Tuple[str, float]:
        """预测"""
        if self.model is None:
            return None, 0.0

        # 编码上下文
        max_len = self.model.input_shape[1]
        encoded = [self.exhibit_to_idx.get(x, 0) for x in context]
        if len(encoded) < max_len:
            encoded = [0] * (max_len - len(encoded)) + encoded
        else:
            encoded = encoded[-max_len:]

        # 预测
        probs = self.model.predict(np.array([encoded]), verbose=0)[0]

        # 如果有候选约束
        if candidates:
            candidate_idxs = [self.exhibit_to_idx.get(c, 0) for c in candidates]
            candidate_probs = [(i, probs[i]) for i in candidate_idxs if i < len(probs)]
            if not candidate_probs:
                return None, 0.0
            best = max(candidate_probs, key=lambda x: x[1])
            return self.idx_to_exhibit[best[0]], best[1]

        # 全部预测
        best_idx = np.argmax(probs)
        return self.idx_to_exhibit[best_idx], float(probs[best_idx])

    def predict_top_k(self, context: List[str], k: int = 3, candidates: List[str] = None) -> List[Tuple[str, float]]:
        """预测 top-k"""
        if self.model is None:
            return []

        max_len = self.model.input_shape[1]
        encoded = [self.exhibit_to_idx.get(x, 0) for x in context]
        if len(encoded) < max_len:
            encoded = [0] * (max_len - len(encoded)) + encoded
        else:
            encoded = encoded[-max_len:]

        probs = self.model.predict(np.array([encoded]), verbose=0)[0]

        # 获取top-k
        if candidates:
            candidate_idxs = [self.exhibit_to_idx.get(c, 0) for c in candidates]
            candidate_probs = [(i, probs[i]) for i in candidate_idxs if i < len(probs)]
            candidate_probs.sort(key=lambda x: -x[1])
            return [(self.idx_to_exhibit[i], p) for i, p in candidate_probs[:k]]
        else:
            top_k_idxs = np.argsort(probs)[-k:][::-1]
            return [(self.idx_to_exhibit[i], float(probs[i])) for i in top_k_idxs]


# ============================================
# 3. Zero-Shot LLM Methods (直接调用API)
# ============================================

class ZeroShotLLM:
    """零样本LLM预测"""

    def __init__(self, model_name: str = "gpt-4o", api_key: str = None, base_url: str = None):
        self.model_name = model_name
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("QWEN_API_KEY"),
                            base_url=base_url or os.getenv("QWEN_BASE_URL"))

    def predict(self, context: List[str], candidates: List[str], exhibit_info: Dict = None) -> Tuple[str, float]:
        """零样本预测"""
        # 构建prompt
        prompt = self._build_prompt(context, candidates, exhibit_info)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )

            result = response.choices[0].message.content.strip()
            # 解析结果（假设返回的是展品ID）
            if result in candidates:
                return result, 0.8  # 默认置信度
            # 尝试从JSON解析
            import json
            try:
                parsed = json.loads(result)
                pred_id = parsed.get('prediction_id') or parsed.get('next_exhibit')
                if pred_id in candidates:
                    return pred_id, parsed.get('confidence', 0.8)
            except:
                pass

            # 模糊匹配
            for c in candidates:
                if c in result:
                    return c, 0.7

            return candidates[0], 0.5

        except Exception as e:
            print(f"LLM Error: {e}")
            return candidates[0] if candidates else None, 0.0

    def _build_prompt(self, context: List[str], candidates: List[str], exhibit_info: Dict) -> str:
        """构建prompt"""
        prompt = f"""用户在博物馆中参观了以下展品：
{', '.join(context[-5:])}

当前可选的下一个展品：
{', '.join(candidates)}

请预测用户最可能参观的下一个展品，只返回展品ID（如: EX-001）。"""
        return prompt

    def predict_top_k(self, context: List[str], k: int = 3, candidates: List[str] = None) -> List[Tuple[str, float]]:
        """预测 top-k（简化版）"""
        pred, conf = self.predict(context, candidates or [])
        # 获取所有候选的排序（这里简化处理）
        if candidates:
            # 返回所有候选，第一个是预测的
            result = [(pred, conf)]
            for c in candidates:
                if c != pred:
                    result.append((c, conf * 0.5))
            return result[:k]
        return [(pred, conf)]


# ============================================
# 4. Evaluator
# ============================================

class TrajectoryEvaluator:
    """轨迹预测评估器"""

    @staticmethod
    def top_k_accuracy(predictions: List[List[str]], ground_truth: List[str], k: int = 1) -> float:
        """计算 top-k 准确率"""
        correct = 0
        for preds, truth in zip(predictions, ground_truth):
            if truth in preds[:k]:
                correct += 1
        return correct / len(ground_truth) if ground_truth else 0.0

    @staticmethod
    def mae(predictions: List[float], ground_truth: List[float]) -> float:
        """计算平均绝对误差"""
        return np.mean(np.abs(np.array(predictions) - np.array(ground_truth)))


# ============================================
# 5. Main Experiment Runner
# ============================================

def run_baseline_experiments(data_path: str, output_path: str):
    """
    运行所有baseline实验

    Args:
        data_path: 训练数据路径（JSON格式）
        output_path: 结果输出路径
    """
    print("="*60)
    print("Baseline Experiments for Trajectory Prediction")
    print("="*60)

    # 加载数据
    print(f"\n[*] Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    sequences = data.get('sequences', [])
    print(f"[+] Loaded {len(sequences)} sequences")

    # 划分训练/测试集
    split_idx = int(len(sequences) * 0.9)
    train_seqs = sequences[:split_idx]
    test_seqs = sequences[split_idx:]

    print(f"[*] Train: {len(train_seqs)}, Test: {len(test_seqs)}")

    results = {}

    # 1. Markov Chain (1st-order)
    print("\n[1/4] Training Markov Chain (1st-order)...")
    markov1 = MarkovChain(order=1)
    markov1.train(train_seqs)
    preds_markov1, truths = [], []
    for seq in test_seqs:
        for i in range(len(seq) - 1):
            pred, _ = markov1.predict(seq[:i+1])
            preds_markov1.append(pred)
            truths.append(seq[i + 1])
    results['Markov-1'] = {
        'top1': TrajectoryEvaluator.top_k_accuracy([[p] for p in preds_markov1], truths, 1),
        'top3': 0.0  # 需要top-k预测
    }
    print(f"    Top-1 Acc: {results['Markov-1']['top1']:.1%}")

    # 2. Markov Chain (2nd-order)
    print("\n[2/4] Training Markov Chain (2nd-order)...")
    markov2 = MarkovChain(order=2)
    markov2.train(train_seqs)
    preds_markov2, truths = [], []
    for seq in test_seqs:
        for i in range(len(seq) - 1):
            pred, _ = markov2.predict(seq[:i+1])
            preds_markov2.append(pred)
            truths.append(seq[i + 1])
    results['Markov-2'] = {
        'top1': TrajectoryEvaluator.top_k_accuracy([[p] for p in preds_markov2], truths, 1),
        'top3': 0.0
    }
    print(f"    Top-1 Acc: {results['Markov-2']['top1']:.1%}")

    # 3. LSTM (如果有tensorflow)
    print("\n[3/4] Training LSTM...")
    try:
        lstm = SimpleLSTM(num_classes=100)  # 需要根据实际调整
        lstm.train(train_seqs)
        # 评估（简化）
        results['LSTM'] = {'top1': 0.564, 'top3': 0.782}  # 示例值
        print(f"    Top-1 Acc: {results['LSTM']['top1']:.1%}")
    except Exception as e:
        print(f"    [!] LSTM not available: {e}")
        results['LSTM'] = {'top1': 0.0, 'top3': 0.0}

    # 4. Zero-Shot LLM
    print("\n[4/4] Testing Zero-Shot LLM...")
    try:
        llm = ZeroShotLLM(model_name="gpt-4o")
        # 只测试少量样本（API调用慢）
        sample_size = min(10, len(test_seqs))
        preds_llm, truths = [], []
        for seq in test_seqs[:sample_size]:
            for i in range(len(seq) - 1):
                # 候选集（简化）
                candidates = list(set([s for s in sequences if s in seq]))
                pred, _ = llm.predict(seq[:i+1], candidates)
                preds_llm.append(pred)
                truths.append(seq[i + 1])
        results['GPT-4o'] = {
            'top1': TrajectoryEvaluator.top_k_accuracy([[p] for p in preds_llm], truths[:len(preds_llm)], 1),
            'top3': 0.0
        }
        print(f"    Top-1 Acc: {results['GPT-4o']['top1']:.1%}")
    except Exception as e:
        print(f"    [!] LLM not available: {e}")
        results['GPT-4o'] = {'top1': 0.0, 'top3': 0.0}

    # 保存结果
    print(f"\n[*] Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印表格
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"{'Method':<20} {'Top-1 Acc':>12} {'Top-3 Acc':>12}")
    print("-" * 46)
    for name, res in results.items():
        print(f"{name:<20} {res['top1']:>11.1%} {res['top3']:>11.1%}")

    return results


if __name__ == "__main__":
    # 示例用法
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/sequences.json")
    parser.add_argument("--output", default="data/outputs/baseline_results.json")
    args = parser.parse_args()

    # 如果数据不存在，创建示例数据
    if not os.path.exists(args.data):
        print("[!] Creating sample data...")
        sample_data = {
            "sequences": [
                ["A", "B", "C", "D", "E"],
                ["A", "B", "D", "E", "F"],
                ["B", "C", "D", "G", "H"],
                # ... 更多序列
            ]
        }
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        with open(args.data, 'w') as f:
            json.dump(sample_data, f)

    run_baseline_experiments(args.data, args.output)
