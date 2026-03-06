#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final baseline runner for the IROS museum-guide paper.

What this script adds compared with the earlier baseline runner:
1) Multi-seed evaluation (e.g., 42,43,44) with mean ± std reporting
2) Separate next-node and dwell-time tables
3) Stronger dwell baselines:
   - NodeMeanDwell
   - DwellGBDT
4) JSON output already structured for paper tables
5) Optional CSV export

Input session format:
[
  {
    "user_id": "sharegpt_000001",
    "sequence": ["展品A", "展品B", "展品C"],
    "durations": [45.0, 30.0, 60.0],
    "attn_levels": [5, 4, 3]
  },
  ...
]

Important:
- This script evaluates strong reproducible baselines only.
- It does NOT run the main Eye-LLM method, because the converted ShareGPT
  sessions currently contain exhibit NAMES rather than topology exhibit IDs.
  If later you provide a name->ID mapping, the main method can be plugged in.
"""

import os
import json
import math
import random
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise ImportError("Please install torch first: pip install torch") from e

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception as e:
    raise ImportError("Please install scikit-learn first: pip install scikit-learn") from e


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_log1p(x: float) -> float:
    return float(np.log1p(max(0.0, x)))

def parse_seeds(seed_text: str) -> List[int]:
    if not seed_text:
        return [42]
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]

def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std(ddof=0)) if len(arr) else 0.0,
    }

def accuracy(pred: List[int], gt: List[int]) -> float:
    if not gt:
        return 0.0
    return float(np.mean([int(p == g) for p, g in zip(pred, gt)]))

def topk_accuracy(logits_or_ranked: List[Any], gt: List[int], k: int = 3) -> float:
    if not gt:
        return 0.0
    correct = 0
    for item, y in zip(logits_or_ranked, gt):
        if isinstance(item, np.ndarray):
            topk = np.argsort(-item)[:k].tolist()
        else:
            topk = item[:k]
        correct += int(y in topk)
    return correct / len(gt)

def mean_reciprocal_rank(logits_or_ranked: List[Any], gt: List[int]) -> float:
    if not gt:
        return 0.0
    rr = []
    for item, y in zip(logits_or_ranked, gt):
        if isinstance(item, np.ndarray):
            ranked = np.argsort(-item).tolist()
        else:
            ranked = item
        if y in ranked:
            rr.append(1.0 / (ranked.index(y) + 1))
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

def pad_left(seq: List[int], max_len: int, pad_value: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_value] * (max_len - len(seq)) + seq

def pad_left_float(seq: List[float], max_len: int, pad_value: float = 0.0) -> List[float]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_value] * (max_len - len(seq)) + seq


class SimpleTopology:
    def __init__(self):
        self.adj: Dict[str, List[str]] = defaultdict(list)

    def build_from_sequences(self, sequences: List[List[str]], undirected: bool = True):
        for seq in sequences:
            for i in range(len(seq) - 1):
                u, v = seq[i], seq[i + 1]
                if v not in self.adj[u]:
                    self.adj[u].append(v)
                if undirected and u not in self.adj[v]:
                    self.adj[v].append(u)

    def neighbors(self, node_id: str) -> List[str]:
        return self.adj.get(node_id, [])


def random_split_by_session(
    sessions: List[Dict],
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    sessions = sessions[:]
    rng.shuffle(sessions)
    n_test = max(1, int(round(len(sessions) * test_ratio)))
    test = sessions[:n_test]
    train = sessions[n_test:]
    return train, test


@dataclass
class Sample:
    user_id: str
    current_node: str
    history_nodes: List[str]
    history_durations: List[float]
    history_attn: List[int]
    next_node: str
    next_duration: float
    candidate_nodes: List[str]
    revisit_count_current: int
    cum_dwell_current: float
    unique_count_so_far: int


class NodeVocab:
    def __init__(self, nodes: List[str]):
        uniq = sorted(list(set(nodes)))
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.nodes = [self.pad_token, self.unk_token] + uniq
        self.stoi = {n: i for i, n in enumerate(self.nodes)}
        self.itos = {i: n for i, n in enumerate(self.nodes)}

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, x: str) -> int:
        return self.stoi.get(x, self.unk_id)

    def decode(self, i: int) -> str:
        return self.itos.get(i, self.unk_token)

    def __len__(self) -> int:
        return len(self.nodes)


def build_samples(
    sessions: List[Dict],
    topology: SimpleTopology,
    history_window: int = 5
) -> List[Sample]:
    samples: List[Sample] = []

    for sess in sessions:
        user_id = sess["user_id"]
        seq = sess["sequence"]
        durs = sess["durations"]
        attn = sess.get("attn_levels", [0] * len(seq))

        if len(seq) < 2:
            continue

        visit_counter = Counter()
        dwell_acc = defaultdict(float)

        for t in range(len(seq) - 1):
            current = seq[t]
            next_node = seq[t + 1]
            next_duration = float(durs[t + 1])

            visit_counter[current] += 1
            dwell_acc[current] += float(durs[t])

            hist_nodes = seq[max(0, t - history_window + 1): t + 1]
            hist_durs = durs[max(0, t - history_window + 1): t + 1]
            hist_attn = attn[max(0, t - history_window + 1): t + 1]

            candidates = topology.neighbors(current)
            if not candidates:
                candidates = list({next_node})

            samples.append(
                Sample(
                    user_id=user_id,
                    current_node=current,
                    history_nodes=hist_nodes,
                    history_durations=[float(x) for x in hist_durs],
                    history_attn=[int(x) for x in hist_attn],
                    next_node=next_node,
                    next_duration=next_duration,
                    candidate_nodes=candidates,
                    revisit_count_current=int(visit_counter[current]),
                    cum_dwell_current=float(dwell_acc[current]),
                    unique_count_so_far=len(set(seq[: t + 1])),
                )
            )
    return samples


class TopologyMarkovBaseline:
    def __init__(self, vocab: NodeVocab):
        self.vocab = vocab
        self.trans_counts = defaultdict(Counter)
        self.global_next = Counter()

    def fit(self, train_samples: List[Sample]):
        for s in train_samples:
            self.trans_counts[s.current_node][s.next_node] += 1
            self.global_next[s.next_node] += 1

    def predict_scores(self, sample: Sample) -> np.ndarray:
        scores = np.full(len(self.vocab), -1e9, dtype=np.float32)
        candidates = sample.candidate_nodes if sample.candidate_nodes else list(self.trans_counts.keys())
        local_counter = self.trans_counts.get(sample.current_node, Counter())

        if local_counter:
            total = sum(local_counter.values())
            for n in candidates:
                scores[self.vocab.encode(n)] = local_counter.get(n, 0) / max(1, total)
        else:
            total = sum(self.global_next.values())
            for n in candidates:
                scores[self.vocab.encode(n)] = self.global_next.get(n, 0) / max(1, total)
        return scores


class NodeMeanDwellBaseline:
    def __init__(self):
        self.node_mean = {}
        self.global_mean = 30.0

    def fit(self, train_samples: List[Sample]):
        bucket = defaultdict(list)
        all_vals = []
        for s in train_samples:
            bucket[s.next_node].append(float(s.next_duration))
            all_vals.append(float(s.next_duration))
        self.node_mean = {k: float(np.mean(v)) for k, v in bucket.items()}
        self.global_mean = float(np.mean(all_vals)) if all_vals else 30.0

    def predict(self, samples: List[Sample]) -> List[float]:
        return [self.node_mean.get(s.next_node, self.global_mean) for s in samples]


class DwellGBDTBaseline:
    def __init__(self, vocab: NodeVocab, history_window: int = 5, random_state: int = 42):
        self.vocab = vocab
        self.history_window = history_window
        self.model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=random_state
        )

    def featurize(self, samples: List[Sample]) -> np.ndarray:
        feats = []
        for s in samples:
            hist_ids = [self.vocab.encode(x) for x in s.history_nodes]
            hist_ids = pad_left(hist_ids, self.history_window, self.vocab.pad_id)

            hist_dur = [safe_log1p(x) for x in s.history_durations]
            hist_dur = pad_left_float(hist_dur, self.history_window, 0.0)

            hist_attn = [float(x) for x in s.history_attn]
            hist_attn = pad_left_float(hist_attn, self.history_window, 0.0)

            row = []
            row.append(self.vocab.encode(s.current_node))
            row.extend(hist_ids)
            row.extend(hist_dur)
            row.extend(hist_attn)
            row.append(float(s.revisit_count_current))
            row.append(safe_log1p(s.cum_dwell_current))
            row.append(float(s.unique_count_so_far))
            row.append(float(len(s.candidate_nodes)))
            feats.append(row)
        return np.asarray(feats, dtype=np.float32)

    def fit(self, train_samples: List[Sample]):
        X = self.featurize(train_samples)
        y = np.array([safe_log1p(s.next_duration) for s in train_samples], dtype=np.float32)
        self.model.fit(X, y)

    def predict(self, samples: List[Sample]) -> List[float]:
        X = self.featurize(samples)
        pred_log = self.model.predict(X)
        return [float(np.expm1(max(0.0, x))) for x in pred_log]


class NextNodeDataset(Dataset):
    def __init__(self, samples: List[Sample], vocab: NodeVocab, history_window: int = 5):
        self.samples = samples
        self.vocab = vocab
        self.history_window = history_window

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        hist_ids = [self.vocab.encode(x) for x in s.history_nodes]
        hist_ids = pad_left(hist_ids, self.history_window, self.vocab.pad_id)

        hist_dur = [safe_log1p(x) for x in s.history_durations]
        hist_dur = pad_left_float(hist_dur, self.history_window, 0.0)

        hist_attn = [float(x) for x in s.history_attn]
        hist_attn = pad_left_float(hist_attn, self.history_window, 0.0)

        candidate_mask = np.zeros(len(self.vocab), dtype=np.float32)
        for n in s.candidate_nodes:
            candidate_mask[self.vocab.encode(n)] = 1.0
        candidate_mask[self.vocab.encode(s.next_node)] = 1.0

        return {
            "hist_nodes": torch.tensor(hist_ids, dtype=torch.long),
            "hist_durs": torch.tensor(hist_dur, dtype=torch.float32),
            "hist_attn": torch.tensor(hist_attn, dtype=torch.float32),
            "label_next": torch.tensor(self.vocab.encode(s.next_node), dtype=torch.long),
            "candidate_mask": torch.tensor(candidate_mask, dtype=torch.float32),
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SeqTransformerMaskedBaseline(nn.Module):
    def __init__(self, num_nodes: int, pad_id: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.pad_id = pad_id
        self.node_emb = nn.Embedding(num_nodes, d_model, padding_idx=pad_id)
        self.dur_proj = nn.Linear(1, d_model)
        self.attn_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_nodes)

    def forward(self, hist_nodes, hist_durs, hist_attn, candidate_mask=None):
        x = self.node_emb(hist_nodes)
        x = x + self.dur_proj(hist_durs.unsqueeze(-1)) + self.attn_proj(hist_attn.unsqueeze(-1))
        x = self.pos_enc(x)
        pad_mask = hist_nodes.eq(self.pad_id)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        pooled = h[:, -1, :]
        logits = self.cls(pooled)
        if candidate_mask is not None:
            logits = logits.masked_fill(candidate_mask <= 0, -1e9)
        return logits


class GraphAwareTransformerBaseline(nn.Module):
    def __init__(self, num_nodes: int, pad_id: int, adj_matrix: np.ndarray, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.pad_id = pad_id
        self.raw_node_emb = nn.Embedding(num_nodes, d_model, padding_idx=pad_id)
        self.self_proj = nn.Linear(d_model, d_model)
        self.neigh_proj = nn.Linear(d_model, d_model)

        A = torch.tensor(adj_matrix, dtype=torch.float32)
        row_sum = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        A = A / row_sum
        self.register_buffer("A_norm", A)

        self.dur_proj = nn.Linear(1, d_model)
        self.attn_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_nodes)

    def graph_enhanced_table(self):
        base = self.raw_node_emb.weight
        neigh = self.A_norm @ base
        return torch.relu(self.self_proj(base) + self.neigh_proj(neigh))

    def forward(self, hist_nodes, hist_durs, hist_attn, candidate_mask=None):
        table = self.graph_enhanced_table()
        x = table[hist_nodes]
        x = x + self.dur_proj(hist_durs.unsqueeze(-1)) + self.attn_proj(hist_attn.unsqueeze(-1))
        x = self.pos_enc(x)
        pad_mask = hist_nodes.eq(self.pad_id)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        pooled = h[:, -1, :]
        logits = self.cls(pooled)
        if candidate_mask is not None:
            logits = logits.masked_fill(candidate_mask <= 0, -1e9)
        return logits


def build_adj_matrix(vocab: NodeVocab, topology: SimpleTopology, add_self_loops: bool = True) -> np.ndarray:
    n = len(vocab)
    A = np.zeros((n, n), dtype=np.float32)
    for node in vocab.nodes:
        if node.startswith("<"):
            continue
        u = vocab.encode(node)
        for nbr in topology.neighbors(node):
            v = vocab.encode(nbr)
            A[u, v] = 1.0
    if add_self_loops:
        for i in range(n):
            A[i, i] = 1.0
    return A


def train_next_model(
    model: nn.Module,
    train_samples: List[Sample],
    val_samples: List[Sample],
    vocab: NodeVocab,
    history_window: int = 5,
    batch_size: int = 64,
    lr: float = 2e-4,
    epochs: int = 20,
    device: str = "cpu"
):
    model = model.to(device)
    train_ds = NextNodeDataset(train_samples, vocab, history_window)
    val_ds = NextNodeDataset(val_samples, vocab, history_window)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_val = -1.0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            hist_nodes = batch["hist_nodes"].to(device)
            hist_durs = batch["hist_durs"].to(device)
            hist_attn = batch["hist_attn"].to(device)
            label_next = batch["label_next"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)

            logits = model(hist_nodes, hist_durs, hist_attn, candidate_mask)
            loss = criterion(logits, label_next)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        pred_all, gt_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                hist_nodes = batch["hist_nodes"].to(device)
                hist_durs = batch["hist_durs"].to(device)
                hist_attn = batch["hist_attn"].to(device)
                label_next = batch["label_next"].to(device)
                candidate_mask = batch["candidate_mask"].to(device)

                logits = model(hist_nodes, hist_durs, hist_attn, candidate_mask)
                pred = torch.argmax(logits, dim=-1)
                pred_all.extend(pred.cpu().tolist())
                gt_all.extend(label_next.cpu().tolist())

        val_acc = accuracy(pred_all, gt_all)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_next_scores_nn(
    model: nn.Module,
    samples: List[Sample],
    vocab: NodeVocab,
    history_window: int = 5,
    batch_size: int = 256,
    device: str = "cpu"
) -> List[np.ndarray]:
    ds = NextNodeDataset(samples, vocab, history_window)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval().to(device)

    outputs = []
    with torch.no_grad():
        for batch in loader:
            hist_nodes = batch["hist_nodes"].to(device)
            hist_durs = batch["hist_durs"].to(device)
            hist_attn = batch["hist_attn"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)

            logits = model(hist_nodes, hist_durs, hist_attn, candidate_mask)
            outputs.extend(logits.cpu().numpy())
    return outputs


def evaluate_next_node(scores: List[np.ndarray], test_samples: List[Sample], vocab: NodeVocab) -> Dict[str, float]:
    gt = [vocab.encode(s.next_node) for s in test_samples]
    pred = [int(np.argmax(x)) for x in scores]
    return {
        "Hit@1": accuracy(pred, gt),
        "Hit@3": topk_accuracy(scores, gt, k=3),
        "MRR": mean_reciprocal_rank(scores, gt),
    }


class StrongBaselineRunner:
    def __init__(self, sessions: List[Dict], history_window: int = 5, seed: int = 42, device: Optional[str] = None):
        set_seed(seed)
        self.sessions = sessions
        self.history_window = history_window
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run_one_seed(self, test_ratio: float = 0.1) -> Dict[str, Dict[str, float]]:
        train_sessions, test_sessions = random_split_by_session(self.sessions, test_ratio=test_ratio, seed=self.seed)

        topology = SimpleTopology()
        topology.build_from_sequences([s["sequence"] for s in train_sessions])

        all_nodes = []
        for s in self.sessions:
            all_nodes.extend(s["sequence"])
        for u in list(topology.adj.keys()):
            all_nodes.append(u)
            all_nodes.extend(topology.adj[u])
        vocab = NodeVocab(all_nodes)

        train_samples = build_samples(train_sessions, topology, self.history_window)
        test_samples = build_samples(test_sessions, topology, self.history_window)
        if len(train_samples) == 0 or len(test_samples) == 0:
            raise ValueError("Empty train/test samples after split.")

        rng = random.Random(self.seed)
        train_perm = train_samples[:]
        rng.shuffle(train_perm)
        n_val = max(1, int(0.1 * len(train_perm)))
        val_samples = train_perm[:n_val]
        fit_samples = train_perm[n_val:]

        out = {"next_node": {}, "dwell": {}}

        markov = TopologyMarkovBaseline(vocab)
        markov.fit(fit_samples)
        markov_scores = [markov.predict_scores(s) for s in test_samples]
        out["next_node"]["TopologyMarkov"] = evaluate_next_node(markov_scores, test_samples, vocab)

        seq_model = SeqTransformerMaskedBaseline(num_nodes=len(vocab), pad_id=vocab.pad_id)
        seq_model = train_next_model(
            seq_model, fit_samples, val_samples, vocab,
            history_window=self.history_window, device=self.device, epochs=20
        )
        seq_scores = predict_next_scores_nn(seq_model, test_samples, vocab, self.history_window, device=self.device)
        out["next_node"]["SeqTransformerMasked"] = evaluate_next_node(seq_scores, test_samples, vocab)

        adj = build_adj_matrix(vocab, topology, add_self_loops=True)
        graph_model = GraphAwareTransformerBaseline(num_nodes=len(vocab), pad_id=vocab.pad_id, adj_matrix=adj)
        graph_model = train_next_model(
            graph_model, fit_samples, val_samples, vocab,
            history_window=self.history_window, device=self.device, epochs=20
        )
        graph_scores = predict_next_scores_nn(graph_model, test_samples, vocab, self.history_window, device=self.device)
        out["next_node"]["GraphAwareTransformer"] = evaluate_next_node(graph_scores, test_samples, vocab)

        gt_dwell = [s.next_duration for s in test_samples]

        node_mean = NodeMeanDwellBaseline()
        node_mean.fit(fit_samples)
        node_mean_pred = node_mean.predict(test_samples)
        out["dwell"]["NodeMeanDwell"] = {
            "MAE": mae(node_mean_pred, gt_dwell),
            "RMSE": rmse(node_mean_pred, gt_dwell)
        }

        dwell_gbdt = DwellGBDTBaseline(vocab=vocab, history_window=self.history_window, random_state=self.seed)
        dwell_gbdt.fit(fit_samples)
        dwell_gbdt_pred = dwell_gbdt.predict(test_samples)
        out["dwell"]["DwellGBDT"] = {
            "MAE": mae(dwell_gbdt_pred, gt_dwell),
            "RMSE": rmse(dwell_gbdt_pred, gt_dwell)
        }

        return out


def aggregate_multiseed(per_seed_results: Dict[int, Dict[str, Dict[str, Dict[str, float]]]]) -> Dict[str, Any]:
    agg = {"next_node": {}, "dwell": {}, "per_seed": per_seed_results}

    methods = set()
    for _, res in per_seed_results.items():
        methods.update(res["next_node"].keys())

    for method in sorted(methods):
        hit1_vals, hit3_vals, mrr_vals = [], [], []
        for _, res in per_seed_results.items():
            if method in res["next_node"]:
                hit1_vals.append(res["next_node"][method]["Hit@1"])
                hit3_vals.append(res["next_node"][method]["Hit@3"])
                mrr_vals.append(res["next_node"][method]["MRR"])
        agg["next_node"][method] = {
            "Hit@1": mean_std(hit1_vals),
            "Hit@3": mean_std(hit3_vals),
            "MRR": mean_std(mrr_vals),
        }

    methods = set()
    for _, res in per_seed_results.items():
        methods.update(res["dwell"].keys())

    for method in sorted(methods):
        mae_vals, rmse_vals = [], []
        for _, res in per_seed_results.items():
            if method in res["dwell"]:
                mae_vals.append(res["dwell"][method]["MAE"])
                rmse_vals.append(res["dwell"][method]["RMSE"])
        agg["dwell"][method] = {
            "MAE": mean_std(mae_vals),
            "RMSE": mean_std(rmse_vals),
        }

    return agg


def fmt(ms: Dict[str, float]) -> str:
    return f"{ms['mean']:.3f} ± {ms['std']:.3f}"

def print_next_node_table(agg: Dict[str, Any]):
    print("\n" + "=" * 96)
    print("Next-node Prediction Baselines (mean ± std over seeds)")
    print("=" * 96)
    print(f"{'Method':<28} {'Hit@1':>18} {'Hit@3':>18} {'MRR':>18}")
    print("-" * 96)
    for method, met in agg["next_node"].items():
        print(
            f"{method:<28} "
            f"{fmt(met['Hit@1']):>18} "
            f"{fmt(met['Hit@3']):>18} "
            f"{fmt(met['MRR']):>18}"
        )


def print_dwell_table(agg: Dict[str, Any]):
    print("\n" + "=" * 96)
    print("Dwell-time Baselines (mean ± std over seeds)")
    print("=" * 96)
    print(f"{'Method':<28} {'MAE':>18} {'RMSE':>18}")
    print("-" * 96)
    for method, met in agg["dwell"].items():
        print(
            f"{method:<28} "
            f"{fmt(met['MAE']):>18} "
            f"{fmt(met['RMSE']):>18}"
        )


def save_csv_tables(agg: Dict[str, Any], output_prefix: str):
    next_csv = output_prefix + "_next_node.csv"
    dwell_csv = output_prefix + "_dwell.csv"

    with open(next_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Hit@1_mean", "Hit@1_std", "Hit@3_mean", "Hit@3_std", "MRR_mean", "MRR_std"])
        for method, met in agg["next_node"].items():
            w.writerow([
                method,
                met["Hit@1"]["mean"], met["Hit@1"]["std"],
                met["Hit@3"]["mean"], met["Hit@3"]["std"],
                met["MRR"]["mean"], met["MRR"]["std"],
            ])

    with open(dwell_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std"])
        for method, met in agg["dwell"].items():
            w.writerow([
                method,
                met["MAE"]["mean"], met["MAE"]["std"],
                met["RMSE"]["mean"], met["RMSE"]["std"],
            ])

    print(f"\nSaved CSV tables to:\n  {next_csv}\n  {dwell_csv}")


def load_sessions(data_path: str) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "sessions" in obj:
        return obj["sessions"]
    raise ValueError("Unsupported data format. Expect list[...] or {'sessions': [...]}.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Final multi-seed baseline runner for IROS museum-guide experiments")
    parser.add_argument("--data", type=str, required=True, help="Path to session JSON")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--history_window", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--output", type=str, default="baseline_results_final.json")
    parser.add_argument("--save_csv", action="store_true")
    args = parser.parse_args()

    sessions = load_sessions(args.data)
    seeds = parse_seeds(args.seeds)

    per_seed = {}
    for seed in seeds:
        print(f"\nRunning seed = {seed}")
        runner = StrongBaselineRunner(
            sessions=sessions,
            history_window=args.history_window,
            seed=seed
        )
        per_seed[seed] = runner.run_one_seed(test_ratio=args.test_ratio)

    agg = aggregate_multiseed(per_seed)
    agg["meta"] = {
        "data": args.data,
        "history_window": args.history_window,
        "test_ratio": args.test_ratio,
        "seeds": seeds,
        "note": "Dwell baselines are based on the provided durations in the converted session file."
    }

    print_next_node_table(agg)
    print_dwell_table(agg)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON to: {args.output}")

    if args.save_csv:
        prefix = os.path.splitext(args.output)[0]
        save_csv_tables(agg, prefix)


if __name__ == "__main__":
    main()
