#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # remove all whitespace
    s = re.sub(r"\s+", "", s)
    # unify common punctuation variants
    trans = str.maketrans({
        "（": "(", "）": ")",
        "【": "[", "】": "]",
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "：": ":", "，": ",", "；": ";",
        "－": "-", "—": "-", "–": "-",
    })
    s = s.translate(trans)
    return s.lower()


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [str(c) for c in df.columns]
    lowered = {c.lower(): c for c in cols}

    name_candidates = [
        c for c in cols
        if c.lower() in {"name", "名称", "展品名称", "node_name", "exhibit_name"}
        or "name" in c.lower()
        or "名称" in c
    ]
    id_candidates = [
        c for c in cols
        if c.lower() in {"id", "编号", "节点id", "node_id", "exhibit_id"}
        or c.lower().endswith("id")
        or c == "ID"
        or "编号" in c
    ]

    if not id_candidates or not name_candidates:
        raise ValueError(f"Could not detect ID/name columns from columns: {cols}")

    return id_candidates[0], name_candidates[0]


def load_mapping_from_excel(excel_path: str, sheet_name=None) -> Dict[str, str]:
    # xls usually needs xlrd installed; xlsx can use openpyxl.
    # pandas will infer engine if possible.
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read Excel file: {excel_path}. If it is .xls, run: pip install xlrd pandas"
        ) from e

    if isinstance(df, dict):
        # if sheet_name=None and workbook has multiple sheets, use the first non-empty one
        for _, sub in df.items():
            if sub is not None and len(sub.columns) > 0 and len(sub) > 0:
                df = sub
                break
        else:
            raise ValueError("No readable sheet found in workbook.")

    id_col, name_col = detect_columns(df)

    mapping = {}
    duplicate_count = 0
    for _, row in df.iterrows():
        raw_id = row.get(id_col)
        raw_name = row.get(name_col)
        if pd.isna(raw_id) or pd.isna(raw_name):
            continue
        exhibit_id = str(raw_id).strip()
        exhibit_name = str(raw_name).strip()
        key = normalize_name(exhibit_name)
        if not key:
            continue
        if key in mapping and mapping[key] != exhibit_id:
            duplicate_count += 1
        mapping[key] = exhibit_id

    print(f"Loaded mapping rows: {len(mapping)}")
    print(f"Detected ID column: {id_col}")
    print(f"Detected name column: {name_col}")
    if duplicate_count:
        print(f"Warning: {duplicate_count} normalized names had duplicate IDs; last one kept.")
    return mapping


def load_sessions(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and 'sessions' in obj:
        return obj['sessions']
    raise ValueError('Unsupported session JSON format.')


def convert_sessions(sessions: List[dict], mapping: Dict[str, str]) -> Tuple[List[dict], List[dict]]:
    converted = []
    unmatched = []

    total_names = 0
    matched_names = 0

    for sess in sessions:
        seq_names = sess.get('sequence', [])
        durations = sess.get('durations', [])
        attn_levels = sess.get('attn_levels', [3] * len(seq_names))

        seq_ids = []
        local_unmatched = []
        for name in seq_names:
            total_names += 1
            key = normalize_name(name)
            exhibit_id = mapping.get(key)
            if exhibit_id is None:
                local_unmatched.append(name)
            else:
                matched_names += 1
                seq_ids.append(exhibit_id)

        # Keep only fully matched sessions for fair evaluation
        if len(local_unmatched) == 0 and len(seq_ids) == len(seq_names) and len(seq_ids) >= 2:
            converted.append({
                'user_id': sess.get('user_id', 'unknown_user'),
                'sequence': seq_ids,
                'sequence_names': seq_names,
                'durations': durations[:len(seq_ids)],
                'attn_levels': attn_levels[:len(seq_ids)],
            })
        else:
            unmatched.append({
                'user_id': sess.get('user_id', 'unknown_user'),
                'unmatched_names': local_unmatched,
                'original_sequence': seq_names,
            })

    coverage = matched_names / max(1, total_names)
    print(f"Total exhibit mentions: {total_names}")
    print(f"Matched exhibit mentions: {matched_names}")
    print(f"Name->ID coverage: {coverage:.2%}")
    print(f"Fully matched sessions kept: {len(converted)}")
    print(f"Sessions with unmatched names: {len(unmatched)}")

    return converted, unmatched


def main():
    parser = argparse.ArgumentParser(description='Build ID-based session JSON from Excel name-ID mapping')
    parser.add_argument('--excel', required=True, help='Path to OS.xls / OS.xlsx')
    parser.add_argument('--sessions', required=True, help='Path to full_sessions_for_baseline.json')
    parser.add_argument('--output', required=True, help='Path to output ID-based sessions JSON')
    parser.add_argument('--unmatched_output', default='', help='Optional path to save unmatched names JSON')
    parser.add_argument('--mapping_output', default='', help='Optional path to save normalized mapping JSON')
    parser.add_argument('--sheet', default=None, help='Optional sheet name')
    args = parser.parse_args()

    mapping = load_mapping_from_excel(args.excel, sheet_name=args.sheet)
    sessions = load_sessions(args.sessions)
    converted, unmatched = convert_sessions(sessions, mapping)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"Saved converted ID sessions to: {out_path}")

    if args.unmatched_output:
        u_path = Path(args.unmatched_output)
        u_path.parent.mkdir(parents=True, exist_ok=True)
        with u_path.open('w', encoding='utf-8') as f:
            json.dump(unmatched, f, ensure_ascii=False, indent=2)
        print(f"Saved unmatched report to: {u_path}")

    if args.mapping_output:
        m_path = Path(args.mapping_output)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        with m_path.open('w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"Saved normalized mapping to: {m_path}")


if __name__ == '__main__':
    main()
