#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the main EyeLLM method on ID-based sessions.

This script fixes two issues in the old comparative script:
1) It loads real session data instead of mock data.
2) It feeds the FULL history (including the current node) into EyeLLMAgent,
   instead of history[:-1].

Important note:
- Hit@1 is always computed.
- Hit@3 / MRR are computed only if EyeLLMAgent.predict_next() returns a ranked list,
  such as top_k_ids / ranked_ids / topk_predictions.
- If your current agent returns only one prediction_id, this script will still run,
  but Hit@3 and MRR will be null. To compare against the baseline table fairly,
  you should expose top-k candidates from the agent.
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# project root: /.../scripts/experiments/run_main_method_eval.py -> project root two levels up
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from agent import EyeLLMAgent  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def parse_seeds(seed_text: str) -> List[int]:
    return [int(x.strip()) for x in seed_text.split(',') if x.strip()]


def random_split_by_session(sessions: List[Dict], test_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    sessions = sessions[:]
    rng.shuffle(sessions)
    n_test = max(1, int(round(len(sessions) * test_ratio)))
    test = sessions[:n_test]
    train = sessions[n_test:]
    return train, test


def load_sessions(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and 'sessions' in obj:
        return obj['sessions']
    raise ValueError('Unsupported session JSON format.')


def attn_num_to_letter(x: Any) -> str:
    mapping = {5: 'A', 4: 'B', 3: 'C', 2: 'D', 1: 'E'}
    try:
        xi = int(x)
        return mapping.get(xi, 'C')
    except Exception:
        if isinstance(x, str) and x.strip().upper() in {'A', 'B', 'C', 'D', 'E'}:
            return x.strip().upper()
        return 'C'


def accuracy(pred: List[str], gt: List[str]) -> float:
    if not gt:
        return 0.0
    return float(np.mean([int(p == g) for p, g in zip(pred, gt)]))


def topk_accuracy(pred_ranked: List[List[str]], gt: List[str], k: int = 3) -> Optional[float]:
    valid = [(p, g) for p, g in zip(pred_ranked, gt) if p is not None]
    if not valid:
        return None
    correct = 0
    for preds, y in valid:
        correct += int(y in preds[:k])
    return correct / len(valid)


def mean_reciprocal_rank(pred_ranked: List[List[str]], gt: List[str]) -> Optional[float]:
    valid = [(p, g) for p, g in zip(pred_ranked, gt) if p is not None]
    if not valid:
        return None
    rr = []
    for preds, y in valid:
        if y in preds:
            rr.append(1.0 / (preds.index(y) + 1))
        else:
            rr.append(0.0)
    return float(np.mean(rr))


def mae(pred: List[float], gt: List[float]) -> float:
    if not gt:
        return 0.0
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(gt))))


def rmse(pred: List[float], gt: List[float]) -> float:
    if not gt:
        return 0.0
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(gt)) ** 2)))


def extract_ranked_ids(prediction: Dict[str, Any]) -> Optional[List[str]]:
    candidates = [
        prediction.get('top_k_ids'),
        prediction.get('ranked_ids'),
        prediction.get('topk_predictions'),
        prediction.get('candidate_ids'),
    ]
    for item in candidates:
        if isinstance(item, list) and item:
            return [str(x) for x in item]

    # if the agent returns a dict of scores, sort descending
    score_dict = prediction.get('candidate_scores')
    if isinstance(score_dict, dict) and score_dict:
        return [k for k, _ in sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)]

    return None


def run_one_seed(
    sessions: List[Dict],
    map_name: str,
    history_window: int,
    test_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    _, test_sessions = random_split_by_session(sessions, test_ratio=test_ratio, seed=seed)

    agent = EyeLLMAgent(map_name=map_name, use_new_architecture=True)

    pred_ids: List[str] = []
    pred_ranked: List[Optional[List[str]]] = []
    gt_ids: List[str] = []
    pred_dur: List[float] = []
    gt_dur: List[float] = []
    skipped_steps = 0

    for sess in test_sessions:
        sequence = sess['sequence']
        durations = sess.get('durations', [30.0] * len(sequence))
        attn_levels = sess.get('attn_levels', [3] * len(sequence))

        if len(sequence) < 2:
            continue

        for i in range(1, len(sequence)):
            history_ids = sequence[max(0, i - history_window): i]
            history_attn = attn_levels[max(0, i - history_window): i]
            ground_truth = sequence[i]
            true_duration = float(durations[i]) if i < len(durations) else 30.0

            if not history_ids:
                skipped_steps += 1
                continue

            agent.prediction_engine.clear_memory()

            ok = True
            for exhibit_id, attn in zip(history_ids, history_attn):
                node_data = agent.topology.query_node(exhibit_id)
                if 'error' in node_data:
                    ok = False
                    break
                name = node_data['info']['name']
                agent.add_observation(exhibit_id, name, attn_num_to_letter(attn))

            if not ok:
                skipped_steps += 1
                continue

            prediction = agent.predict_next(verbose=False)
            pred_id = str(prediction.get('prediction_id', ''))
            est_duration = float(prediction.get('estimated_duration', 30.0))
            ranked_ids = extract_ranked_ids(prediction)

            pred_ids.append(pred_id)
            pred_ranked.append(ranked_ids)
            gt_ids.append(str(ground_truth))
            pred_dur.append(est_duration)
            gt_dur.append(true_duration)

    result = {
        'Hit@1': accuracy(pred_ids, gt_ids),
        'Hit@3': topk_accuracy(pred_ranked, gt_ids, k=3),
        'MRR': mean_reciprocal_rank(pred_ranked, gt_ids),
        'MAE': mae(pred_dur, gt_dur),
        'RMSE': rmse(pred_dur, gt_dur),
        'num_eval_steps': len(gt_ids),
        'skipped_steps': skipped_steps,
    }
    return result


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {'mean': float(arr.mean()), 'std': float(arr.std(ddof=0))}


def aggregate(per_seed: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    out = {'per_seed': per_seed}
    keys = ['Hit@1', 'Hit@3', 'MRR', 'MAE', 'RMSE']
    summary = {}
    for k in keys:
        vals = [res[k] for _, res in per_seed.items() if res.get(k) is not None]
        summary[k] = mean_std(vals) if vals else None
    summary['num_eval_steps'] = {str(seed): res['num_eval_steps'] for seed, res in per_seed.items()}
    summary['skipped_steps'] = {str(seed): res['skipped_steps'] for seed, res in per_seed.items()}
    out['summary'] = summary
    return out


def print_summary(agg: Dict[str, Any]):
    s = agg['summary']
    print('\n' + '=' * 88)
    print('Main Method (EyeLLM) Results')
    print('=' * 88)
    if s['Hit@1'] is not None:
        print(f"Hit@1 : {s['Hit@1']['mean']:.3f} ± {s['Hit@1']['std']:.3f}")
    if s['Hit@3'] is not None:
        print(f"Hit@3 : {s['Hit@3']['mean']:.3f} ± {s['Hit@3']['std']:.3f}")
    else:
        print('Hit@3 : unavailable (agent did not expose ranked predictions)')
    if s['MRR'] is not None:
        print(f"MRR   : {s['MRR']['mean']:.3f} ± {s['MRR']['std']:.3f}")
    else:
        print('MRR   : unavailable (agent did not expose ranked predictions)')
    if s['MAE'] is not None:
        print(f"MAE   : {s['MAE']['mean']:.3f} ± {s['MAE']['std']:.3f}")
    if s['RMSE'] is not None:
        print(f"RMSE  : {s['RMSE']['mean']:.3f} ± {s['RMSE']['std']:.3f}")
    print(f"num_eval_steps by seed: {s['num_eval_steps']}")
    print(f"skipped_steps by seed : {s['skipped_steps']}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate main EyeLLM method on ID-based sessions')
    parser.add_argument('--data', required=True, help='Path to ID-based sessions JSON')
    parser.add_argument('--map', default='TH', help='Map name used by EyeLLMAgent')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--history_window', type=int, default=5)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    sessions = load_sessions(args.data)
    seeds = parse_seeds(args.seeds)

    per_seed = {}
    for seed in seeds:
        print(f'Running main method seed = {seed}')
        per_seed[seed] = run_one_seed(
            sessions=sessions,
            map_name=args.map,
            history_window=args.history_window,
            test_ratio=args.test_ratio,
            seed=seed,
        )

    agg = aggregate(per_seed)
    agg['meta'] = {
        'data': args.data,
        'map': args.map,
        'test_ratio': args.test_ratio,
        'history_window': args.history_window,
        'seeds': seeds,
        'note': 'Hit@3/MRR require ranked outputs from EyeLLMAgent.predict_next().'
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    print_summary(agg)
    print(f'\nSaved main-method results to: {out_path}')


if __name__ == '__main__':
    main()
