#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert ShareGPT-style museum JSONL into baseline session JSON.

Supported source sample structure:
[
  {
    "conversations": [
      {"from": "human", "value": "```json ... ```"},
      {"from": "gpt",   "value": "```json ... ```"}
    ]
  },
  ...
]

Target output format:
[
  {
    "user_id": "sharegpt_000001",
    "sequence": ["展品A", "展品B", "展品C"],
    "durations": [45.0, 30.0, 60.0],
    "attn_levels": [5, 4, 3]
  }
]

Notes:
- Only `plan_scan_path` is converted into full sessions by default.
- `predict_next` is ignored unless it contains a non-empty history.
- `attribution` is ignored.
- Since the source file usually lacks real timestamps, durations here are pseudo durations
  derived from attention level or a constant fallback. They are OK for debugging / preliminary
  runs, but NOT recommended as the final dwell-time baseline reported in the paper.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ATTN_TO_INT = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "E": 1,
}

ATTN_TO_PSEUDO_DURATION = {
    "A": 60.0,
    "B": 45.0,
    "C": 30.0,
    "D": 20.0,
    "E": 12.0,
}


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def loads_embedded_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(strip_code_fence(text))
    except Exception:
        return None


def get_turn(conversations: List[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    for turn in conversations:
        if turn.get("from") == role:
            return turn
    return None


def convert_plan_scan_path(
    src_item: Dict[str, Any],
    sample_idx: int,
    duration_mode: str = "attention"
) -> Optional[Dict[str, Any]]:
    convs = src_item.get("conversations", [])
    human = get_turn(convs, "human")
    gpt = get_turn(convs, "gpt")
    if not human or not gpt:
        return None

    h = loads_embedded_json(human.get("value", ""))
    y = loads_embedded_json(gpt.get("value", ""))
    if not h or not y:
        return None

    if h.get("task") != "plan_scan_path":
        return None

    scan_path = y.get("scan_path", [])
    if not isinstance(scan_path, list) or len(scan_path) < 2:
        return None

    sequence = []
    attn_levels = []
    durations = []

    for step in scan_path:
        name = step.get("name")
        attn = step.get("attention_level", "C")
        if not name:
            continue

        sequence.append(name)
        attn_levels.append(ATTN_TO_INT.get(attn, 3))

        if duration_mode == "attention":
            durations.append(ATTN_TO_PSEUDO_DURATION.get(attn, 30.0))
        else:
            durations.append(30.0)

    if len(sequence) < 2:
        return None

    return {
        "user_id": f"sharegpt_{sample_idx:06d}",
        "sequence": sequence,
        "durations": durations,
        "attn_levels": attn_levels,
        "source_task": "plan_scan_path",
    }


def convert_predict_next_with_history(
    src_item: Dict[str, Any],
    sample_idx: int,
    duration_mode: str = "attention"
) -> Optional[Dict[str, Any]]:
    """
    Optional conversion:
    If a predict_next sample contains non-empty history, we can convert it into
    one short session = history + predicted node. Most shown samples have empty history.
    """
    convs = src_item.get("conversations", [])
    human = get_turn(convs, "human")
    gpt = get_turn(convs, "gpt")
    if not human or not gpt:
        return None

    h = loads_embedded_json(human.get("value", ""))
    y = loads_embedded_json(gpt.get("value", ""))
    if not h or not y:
        return None

    if h.get("task") != "predict_next":
        return None

    history = h.get("history", [])
    pred = y.get("prediction", {})
    pred_name = pred.get("name")
    pred_attn = pred.get("attention_level", "C")

    if not history or not pred_name:
        return None

    sequence = list(history) + [pred_name]
    attn_levels = [3] * len(history) + [ATTN_TO_INT.get(pred_attn, 3)]

    if duration_mode == "attention":
        durations = [30.0] * len(history) + [ATTN_TO_PSEUDO_DURATION.get(pred_attn, 30.0)]
    else:
        durations = [30.0] * len(sequence)

    if len(sequence) < 2:
        return None

    return {
        "user_id": f"sharegpt_pred_{sample_idx:06d}",
        "sequence": sequence,
        "durations": durations,
        "attn_levels": attn_levels,
        "source_task": "predict_next",
    }


def convert_file(
    input_path: Path,
    output_path: Path,
    duration_mode: str = "attention",
    include_predict_next_history: bool = False,
):
    sessions = []
    stats = {
        "total_rows": 0,
        "plan_scan_path_used": 0,
        "predict_next_used": 0,
        "skipped": 0,
    }

    with input_path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
            iterator = data
        else:
            iterator = [json.loads(line) for line in f if line.strip()]

    for idx, item in enumerate(iterator):
        stats["total_rows"] += 1

        sess = convert_plan_scan_path(item, idx, duration_mode=duration_mode)
        if sess is not None:
            sessions.append(sess)
            stats["plan_scan_path_used"] += 1
            continue

        if include_predict_next_history:
            sess = convert_predict_next_with_history(item, idx, duration_mode=duration_mode)
            if sess is not None:
                sessions.append(sess)
                stats["predict_next_used"] += 1
                continue

        stats["skipped"] += 1

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

    print("Saved:", output_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to sharegpt json/jsonl")
    parser.add_argument("--output", required=True, help="Output path for converted sessions json")
    parser.add_argument(
        "--duration_mode",
        default="attention",
        choices=["attention", "constant"],
        help="How to create durations when true timestamps are missing"
    )
    parser.add_argument(
        "--include_predict_next_history",
        action="store_true",
        help="Also convert predict_next samples with non-empty history"
    )
    args = parser.parse_args()

    convert_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        duration_mode=args.duration_mode,
        include_predict_next_history=args.include_predict_next_history,
    )


if __name__ == "__main__":
    main()
