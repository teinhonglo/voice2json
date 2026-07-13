#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a safe voice2json grammar from MAC-SLU training queries.

Each unique normalized training query becomes one synthetic voice2json intent.
The synthetic intent is mapped externally to the complete MAC-SLU semantics
list, which allows multi-intent outputs even though voice2json itself emits
one intent object per recognized utterance.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def normalize_for_voice2json(text: str) -> str:
    """Normalize and tokenize Chinese text for a plain JSGF sentence.

    Chinese characters are separated by spaces. Contiguous ASCII letters and
    digits are retained as one token. Grammar metacharacters and punctuation
    are removed, preventing invalid OpenGrm/FST topology.
    """
    text = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    tokens: List[str] = []
    ascii_buffer: List[str] = []

    def flush_ascii() -> None:
        if ascii_buffer:
            tokens.append("".join(ascii_buffer))
            ascii_buffer.clear()

    for ch in text:
        if _CJK_RE.fullmatch(ch):
            flush_ascii()
            tokens.append(ch)
        elif ch.isascii() and (ch.isalnum() or ch == "'"):
            ascii_buffer.append(ch)
        else:
            flush_ascii()

    flush_ascii()
    return " ".join(tokens)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL: {path}:{lineno}") from exc
    return rows


def canonical_semantics(value: Any) -> Tuple[str, List[dict]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []

    if not isinstance(value, list):
        value = []

    signature = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return signature, value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--grammar-out", required=True)
    p.add_argument("--intent-map-out", required=True)
    p.add_argument("--stats-out", required=True)
    p.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="0 means all unique training queries",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_jsonl)
    grammar_path = Path(args.grammar_out)
    map_path = Path(args.intent_map_out)
    stats_path = Path(args.stats_out)

    rows = load_jsonl(train_path)

    # normalized query -> semantic signature counts
    query_semantic_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    signature_to_semantics: Dict[str, List[dict]] = {}
    query_raw_examples: Dict[str, str] = {}

    skipped_empty = 0
    for row in rows:
        raw_query = str(row.get("query") or row.get("asr_text") or "")
        norm_query = normalize_for_voice2json(raw_query)
        if not norm_query:
            skipped_empty += 1
            continue

        signature, semantics = canonical_semantics(row.get("semantics", []))
        query_semantic_counts[norm_query][signature] += 1
        signature_to_semantics[signature] = semantics
        query_raw_examples.setdefault(norm_query, raw_query)

    items = sorted(query_semantic_counts.items(), key=lambda x: x[0])
    if args.max_entries > 0:
        items = items[: args.max_entries]

    grammar_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    intent_map: Dict[str, dict] = {}
    conflict_count = 0

    with grammar_path.open("w", encoding="utf-8") as grammar_f:
        grammar_f.write(
            "# Auto-generated from MAC-SLU train.jsonl.\n"
            "# Plain sentences only: no optional blocks or epsilon paths.\n"
            "# Multi-intent semantics are stored in macslu_intent_map.json.\n\n"
        )

        for index, (norm_query, semantic_counter) in enumerate(items, start=1):
            intent_name = f"MACSLU_{index:07d}"
            best_signature, best_count = semantic_counter.most_common(1)[0]
            if len(semantic_counter) > 1:
                conflict_count += 1

            grammar_f.write(f"[{intent_name}]\n")
            grammar_f.write(f"{norm_query}\n\n")

            intent_map[intent_name] = {
                "query": query_raw_examples[norm_query],
                "normalized_query": norm_query,
                "semantics": signature_to_semantics[best_signature],
                "training_count": sum(semantic_counter.values()),
                "selected_label_count": best_count,
                "num_conflicting_labels": len(semantic_counter),
            }

    map_path.write_text(
        json.dumps(intent_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "input_rows": len(rows),
        "unique_normalized_queries": len(query_semantic_counts),
        "grammar_entries": len(items),
        "queries_with_conflicting_labels": conflict_count,
        "skipped_empty_queries": skipped_empty,
        "grammar_file": str(grammar_path),
        "intent_map_file": str(map_path),
    }
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
