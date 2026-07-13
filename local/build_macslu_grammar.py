#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a safe voice2json grammar from MAC-SLU single-intent queries.

Each grammar section is a MAC-SLU domain/intent label, similar to
conf/grammar_zh.ini where one intent section contains multiple sentence
patterns. Training rows with zero or multiple semantic intents are excluded
from grammar training only; test JSONL rows are still evaluated by the
inference/evaluation stages.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_DICT_VARIANT_RE = re.compile(r"\(\d+\)$")
_SECTION_UNSAFE_RE = re.compile(r"[\[\]\r\n]")


def normalize_dictionary_word(word: str) -> str:
    """Normalize a dictionary word for membership checks."""
    word = _DICT_VARIANT_RE.sub("", word)
    return unicodedata.normalize("NFKC", word).strip().lower()


def load_dictionary_words(paths: Iterable[Path]) -> Set[str]:
    """Load known words from pronunciation dictionary files."""
    words: Set[str] = set()
    for path in paths:
        if not path.is_file():
            continue

        with path.open("r", encoding="utf-8") as dict_file:
            for line in dict_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                word = normalize_dictionary_word(line.split(maxsplit=1)[0])
                if word:
                    words.add(word)

    return words


def normalize_for_voice2json(text: str, known_words: Set[str]) -> str:
    """Normalize and tokenize Chinese text for a plain JSGF sentence.

    Chinese characters are separated by spaces. Contiguous ASCII letters and
    digits are retained as one token. Grammar metacharacters and punctuation
    are removed, preventing invalid OpenGrm/FST topology. Tokens outside of
    the pronunciation dictionaries are removed so profile training does not
    have to guess OOV pronunciations with G2P.
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
    tokens = [
        token for token in tokens if normalize_dictionary_word(token) in known_words
    ]

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


def section_name_for_semantic(semantic: dict) -> str:
    domain = str(semantic.get("domain") or "UNKNOWN_DOMAIN")
    intent = str(semantic.get("intent") or "UNKNOWN_INTENT")
    name = f"MACSLU_{domain}_{intent}"
    name = _SECTION_UNSAFE_RE.sub("_", unicodedata.normalize("NFKC", name))
    name = re.sub(r"\s+", "_", name.strip())
    return name or "MACSLU_UNKNOWN"


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
        help=(
            "0 means all unique single-intent/query pairs. When positive, "
            "keeps the first N pairs in train JSONL order."
        ),
    )
    p.add_argument(
        "--dictionary",
        action="append",
        required=True,
        help=(
            "Pronunciation dictionary to use as a known-word allowlist. May be "
            "passed multiple times. OOV tokens are removed from normalized "
            "sentences."
        ),
    )
    p.add_argument(
        "--dictionary",
        action="append",
        required=True,
        help=(
            "Pronunciation dictionary to use as a known-word allowlist. May be "
            "passed multiple times. OOV tokens are removed from normalized "
            "sentences."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_jsonl)
    grammar_path = Path(args.grammar_out)
    map_path = Path(args.intent_map_out)
    stats_path = Path(args.stats_out)

    rows = load_jsonl(train_path)
    known_words = load_dictionary_words(Path(p) for p in args.dictionary)
    if not known_words:
        raise ValueError(
            "No known words were loaded from --dictionary; cannot build an "
            "OOV-filtered MAC-SLU grammar."
        )

    sections: Dict[str, dict] = {}
    normalized_query_sections: Dict[str, Set[str]] = {}

    skipped_empty_query = 0
    zero_intent_rows_excluded = 0
    multi_intent_rows_excluded = 0
    grammar_entries = 0

    for row in rows:
        raw_query = str(row.get("query") or row.get("asr_text") or "")
        norm_query = normalize_for_voice2json(raw_query, known_words=known_words)
        if not norm_query:
            skipped_empty_query += 1
            continue

        _, semantics = canonical_semantics(row.get("semantics", []))
        if len(semantics) == 0:
            zero_intent_rows_excluded += 1
            continue
        if len(semantics) > 1:
            multi_intent_rows_excluded += 1
            continue

        semantic = semantics[0]
        section_name = section_name_for_semantic(semantic)
        if section_name not in sections:
            sections[section_name] = {
                "domain": semantic.get("domain", ""),
                "intent": semantic.get("intent", ""),
                "queries": {},
                "training_count": 0,
            }

        section = sections[section_name]
        section["training_count"] += 1
        normalized_query_sections.setdefault(norm_query, set()).add(section_name)

        if norm_query in section["queries"]:
            continue

        if (args.max_entries > 0) and (grammar_entries >= args.max_entries):
            continue

        section["queries"][norm_query] = {
            "query": raw_query,
            "normalized_query": norm_query,
            "semantics": semantics,
        }
        grammar_entries += 1

    grammar_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    intent_map: Dict[str, dict] = {}

    with grammar_path.open("w", encoding="utf-8") as grammar_f:
        grammar_f.write(
            "# Auto-generated from MAC-SLU train.jsonl.\n"
            "# Plain single-intent sentences only.\n"
            "# Section names are MAC-SLU domain/intent labels.\n"
            "# Per-query slot semantics are stored in macslu_intent_map.json.\n\n"
        )

        for section_name, section in sections.items():
            if not section["queries"]:
                continue

            grammar_f.write(f"[{section_name}]\n")
            for norm_query in section["queries"]:
                grammar_f.write(f"{norm_query}\n")
            grammar_f.write("\n")

            query_entries = list(section["queries"].values())
            intent_map[section_name] = {
                "domain": section["domain"],
                "intent": section["intent"],
                "queries": query_entries,
                "query_semantics": {
                    entry["normalized_query"]: entry["semantics"]
                    for entry in query_entries
                },
                "training_count": section["training_count"],
                "grammar_entries": len(query_entries),
            }

    map_path.write_text(
        json.dumps(intent_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ambiguous_normalized_queries = sum(
        1 for section_names in normalized_query_sections.values() if len(section_names) > 1
    )
    sections_with_entries = sum(1 for section in sections.values() if section["queries"])
    stats = {
        "input_rows": len(rows),
        "single_intent_sections": len(sections),
        "single_intent_sections_with_entries": sections_with_entries,
        "unique_normalized_queries": len(normalized_query_sections),
        "grammar_entries": grammar_entries,
        "ambiguous_normalized_queries": ambiguous_normalized_queries,
        "skipped_empty_queries": skipped_empty_query,
        "zero_intent_rows_excluded_from_grammar": zero_intent_rows_excluded,
        "multi_intent_rows_excluded_from_grammar": multi_intent_rows_excluded,
        "dictionary_files": args.dictionary,
        "known_words": len(known_words),
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
