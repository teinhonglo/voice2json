#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a voice2json grammar from the MAC-SLU training split.

The generated INI follows the same structure as ``conf/grammar_zh.ini``:
intent sections contain sentence patterns, while referenced slot sections
enumerate all observed values. Unlike the previous implementation, this
builder keeps zero-, single-, and multi-intent rows. A synthetic voice2json
intent represents the ordered MAC-SLU intent sequence in each utterance.

Explicit semantic slots are converted into reusable grammar slots. Values of
``操作``/action-like slots are classified as verbs; all other semantic values
are classified as nouns. If jieba.posseg is installed, additional uncovered
nouns and verbs are converted into intent-specific lexical slot sections, so
every generated noun/verb section is referenced by at least one pattern.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Sequence, Set, Tuple

try:
    import jieba.posseg as pseg  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pseg = None

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_DICT_VARIANT_RE = re.compile(r"\(\d+\)$")
_PLACEHOLDER_RE = re.compile(r"<[^<>]+>")
_SECTION_UNSAFE_RE = re.compile(r"[^0-9A-Za-z_\u3400-\u4dbf\u4e00-\u9fff]+")
_ACTION_SLOT_NAMES = {
    "操作",
    "动作",
    "動作",
    "action",
    "operation",
    "操作_concrete",
    "__act__",
}


def normalize_dictionary_word(word: str) -> str:
    word = _DICT_VARIANT_RE.sub("", word)
    return unicodedata.normalize("NFKC", word).strip().lower()


def load_dictionary_words(paths: Iterable[Path]) -> Set[str]:
    words: Set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as dictionary_file:
            for line in dictionary_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                word = normalize_dictionary_word(line.split(maxsplit=1)[0])
                if word:
                    words.add(word)
    return words


def _tokenize_text(text: str) -> List[str]:
    text = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    tokens: List[str] = []
    ascii_buffer: List[str] = []

    def flush_ascii() -> None:
        if ascii_buffer:
            tokens.append("".join(ascii_buffer))
            ascii_buffer.clear()

    for char in text:
        if _CJK_RE.fullmatch(char):
            flush_ascii()
            tokens.append(char)
        elif char.isascii() and (char.isalnum() or char == "'"):
            ascii_buffer.append(char)
        else:
            flush_ascii()
    flush_ascii()
    return tokens


def normalize_for_voice2json(text: str, known_words: Set[str]) -> str:
    """Normalize text to the token form accepted by the Mandarin profile."""
    tokens = _tokenize_text(text)
    if known_words:
        tokens = [
            token
            for token in tokens
            if normalize_dictionary_word(token) in known_words
        ]
    return " ".join(tokens)


def normalize_pattern_for_voice2json(pattern: str, known_words: Set[str]) -> str:
    """Normalize literal text while preserving ``<slot>`` placeholders."""
    output: List[str] = []
    cursor = 0
    for match in _PLACEHOLDER_RE.finditer(pattern):
        literal = normalize_for_voice2json(pattern[cursor : match.start()], known_words)
        if literal:
            output.append(literal)
        output.append(match.group(0))
        cursor = match.end()
    literal = normalize_for_voice2json(pattern[cursor:], known_words)
    if literal:
        output.append(literal)
    return " ".join(output)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL: {path}:{line_number}") from exc
    return rows


def canonical_semantics(value: Any) -> List[dict]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []

    semantics: List[dict] = []
    for semantic in value:
        if not isinstance(semantic, dict):
            continue
        slots = semantic.get("slots")
        implicit_slots = semantic.get("implicit_slots")
        semantics.append(
            {
                "domain": str(semantic.get("domain") or ""),
                "intent": str(semantic.get("intent") or ""),
                "slots": slots if isinstance(slots, dict) else {},
                "implicit_slots": (
                    implicit_slots if isinstance(implicit_slots, dict) else {}
                ),
            }
        )
    return semantics


def _safe_fragment(value: str, fallback: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).strip()
    value = _SECTION_UNSAFE_RE.sub("_", value).strip("_")
    return value or fallback


def intent_signature(semantics: Sequence[dict]) -> List[dict]:
    return [
        {
            "domain": str(item.get("domain") or ""),
            "intent": str(item.get("intent") or ""),
        }
        for item in semantics
    ]


def section_name_for_signature(signature: Sequence[dict]) -> str:
    if not signature:
        return "MACSLU_NO_INTENT"
    readable = "__".join(
        f"{_safe_fragment(item.get('domain', ''), 'UNKNOWN_DOMAIN')}_"
        f"{_safe_fragment(item.get('intent', ''), 'UNKNOWN_INTENT')}"
        for item in signature
    )
    digest = hashlib.sha1(
        json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    readable = readable[:120].rstrip("_")
    return f"MACSLU_{readable}_{digest}"


def slot_section_name(
    frame_index: int,
    domain: str,
    intent: str,
    slot_name: str,
    kind: str,
) -> str:
    raw = f"MACSLU_F{frame_index + 1}_{domain}_{intent}_{slot_name}_{kind}"
    safe = _safe_fragment(raw, f"MACSLU_F{frame_index + 1}_{kind}")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:130].rstrip('_')}_{digest}"


def pos_slot_section_name(intent_section: str, kind: str) -> str:
    raw = f"{intent_section}_POS_{kind.upper()}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{_safe_fragment(raw, 'MACSLU_POS')[:140].rstrip('_')}_{digest}"


def slot_kind(slot_name: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(slot_name or "")).strip().lower()
    if (
        normalized in {name.lower() for name in _ACTION_SLOT_NAMES}
        or "操作" in normalized
        or "动作" in normalized
        or "動作" in normalized
        or "action" in normalized
        or "operation" in normalized
    ):
        return "verb"
    return "noun"


def iter_scalar_values(value: Any) -> Iterator[str]:
    if value is None:
        return
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        if text:
            yield text
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_scalar_values(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_scalar_values(item)


def semantic_slot_occurrences(semantics: Sequence[dict]) -> List[dict]:
    occurrences: List[dict] = []
    for frame_index, semantic in enumerate(semantics):
        domain = str(semantic.get("domain") or "")
        intent = str(semantic.get("intent") or "")
        slots = semantic.get("slots")
        if not isinstance(slots, dict):
            continue
        for name, raw_value in slots.items():
            for value in iter_scalar_values(raw_value):
                kind = slot_kind(str(name))
                section = slot_section_name(
                    frame_index,
                    domain,
                    intent,
                    str(name),
                    kind,
                )
                occurrences.append(
                    {
                        "frame_index": frame_index,
                        "domain": domain,
                        "intent": intent,
                        "slot_name": str(name),
                        "kind": kind,
                        "value": value,
                        "section": section,
                    }
                )
    return occurrences


def replace_explicit_slots(query: str, occurrences: Sequence[dict]) -> Tuple[str, List[dict]]:
    """Replace longest non-overlapping values without touching placeholders."""
    text = unicodedata.normalize("NFKC", str(query or ""))
    occupied = [False] * len(text)
    spans: List[Tuple[int, int, dict]] = []

    # Resolve spans on the original text. This prevents later values such as
    # "音乐" from matching inside an already inserted slot-section name.
    for occurrence in sorted(
        occurrences,
        key=lambda item: (len(str(item["value"])), -int(item["frame_index"])),
        reverse=True,
    ):
        value = str(occurrence["value"])
        if not value:
            continue
        start = 0
        while True:
            index = text.find(value, start)
            if index < 0:
                break
            end = index + len(value)
            if not any(occupied[index:end]):
                for position in range(index, end):
                    occupied[position] = True
                spans.append((index, end, dict(occurrence)))
                break
            start = index + 1

    if not spans:
        return text, []

    output: List[str] = []
    cursor = 0
    used: List[dict] = []
    for start, end, occurrence in sorted(spans, key=lambda item: item[0]):
        output.append(text[cursor:start])
        output.append(f"<{occurrence['section']}>")
        cursor = end
        used.append(occurrence)
    output.append(text[cursor:])
    return "".join(output), used


def extract_pos_terms(query: str) -> Tuple[Set[str], Set[str]]:
    nouns: Set[str] = set()
    verbs: Set[str] = set()
    if pseg is None:
        return nouns, verbs
    for pair in pseg.cut(str(query or "")):
        word = str(pair.word).strip()
        flag = str(pair.flag or "")
        if not word:
            continue
        if flag.startswith("v"):
            verbs.add(word)
        elif flag.startswith("n"):
            nouns.add(word)
    return nouns, verbs


def ordered_add(mapping: MutableMapping[str, None], value: str) -> None:
    if value:
        mapping.setdefault(value, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--grammar-out", required=True)
    parser.add_argument("--intent-map-out", required=True)
    parser.add_argument("--stats-out", required=True)
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="0 keeps all unique training patterns; positive values keep the first N.",
    )
    parser.add_argument(
        "--dictionary",
        action="append",
        required=True,
        help="Pronunciation dictionary allowlist. May be supplied multiple times.",
    )
    parser.add_argument(
        "--disable-pos-lexicon",
        action="store_true",
        help="Do not collect the optional jieba noun/verb lexicons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_jsonl)
    grammar_path = Path(args.grammar_out)
    map_path = Path(args.intent_map_out)
    stats_path = Path(args.stats_out)

    rows = load_jsonl(train_path)
    known_words = load_dictionary_words(Path(path) for path in args.dictionary)
    if not known_words:
        raise ValueError("No known words were loaded from --dictionary")

    sections: "OrderedDict[str, dict]" = OrderedDict()
    slot_values: "OrderedDict[str, OrderedDict[str, None]]" = OrderedDict()
    entity_metadata: Dict[str, dict] = {}
    global_nouns: "OrderedDict[str, None]" = OrderedDict()
    global_verbs: "OrderedDict[str, None]" = OrderedDict()
    normalized_pattern_sections: Dict[str, Set[str]] = {}

    empty_queries = 0
    grammar_entries = 0
    zero_intent_rows = 0
    single_intent_rows = 0
    multi_intent_rows = 0

    for row in rows:
        raw_query = str(row.get("query") or row.get("asr_text") or "").strip()
        normalized_query = normalize_for_voice2json(raw_query, known_words)
        if not normalized_query:
            empty_queries += 1
            continue

        semantics = canonical_semantics(row.get("semantics", []))
        if not semantics:
            zero_intent_rows += 1
        elif len(semantics) == 1:
            single_intent_rows += 1
        else:
            multi_intent_rows += 1

        signature = intent_signature(semantics)
        section_name = section_name_for_signature(signature)
        section = sections.setdefault(
            section_name,
            {
                "intent_signature": signature,
                "patterns": OrderedDict(),
                "queries": [],
                "entity_map": {},
                "training_count": 0,
            },
        )
        section["training_count"] += 1

        occurrences = semantic_slot_occurrences(semantics)

        # Add POS-derived lexical slots that are not already covered by an
        # explicit semantic value. These slots improve lexical coverage while
        # remaining tied to the same synthetic intent section. They are not
        # mapped back into MAC-SLU semantics during inference.
        if not args.disable_pos_lexicon:
            pos_nouns, pos_verbs = extract_pos_terms(raw_query)
            explicit_values = [str(item["value"]) for item in occurrences]
            for kind, terms in (("noun", pos_nouns), ("verb", pos_verbs)):
                for term in sorted(terms, key=len, reverse=True):
                    if not term or any(term in value or value in term for value in explicit_values):
                        continue
                    occurrences.append(
                        {
                            "frame_index": -1,
                            "domain": "",
                            "intent": "",
                            "slot_name": f"__pos_{kind}__",
                            "kind": kind,
                            "value": term,
                            "section": pos_slot_section_name(section_name, kind),
                            "map_to_semantics": False,
                        }
                    )

        raw_pattern, used_occurrences = replace_explicit_slots(raw_query, occurrences)
        normalized_pattern = normalize_pattern_for_voice2json(raw_pattern, known_words)
        if not normalized_pattern:
            normalized_pattern = normalized_query

        is_new_pattern = normalized_pattern not in section["patterns"]
        if is_new_pattern and args.max_entries > 0 and grammar_entries >= args.max_entries:
            # Skip before collecting slot values so a bounded debug grammar
            # never contains slot sections that no retained pattern references.
            continue

        for occurrence in occurrences:
            normalized_value = normalize_for_voice2json(
                str(occurrence["value"]), known_words
            )
            if not normalized_value:
                continue
            values = slot_values.setdefault(occurrence["section"], OrderedDict())
            ordered_add(values, normalized_value)
            entity_metadata[occurrence["section"]] = {
                "frame_index": occurrence["frame_index"],
                "domain": occurrence["domain"],
                "intent": occurrence["intent"],
                "slot_name": occurrence["slot_name"],
                "kind": occurrence["kind"],
            }
            section["entity_map"][occurrence["section"]] = entity_metadata[
                occurrence["section"]
            ]
            if occurrence["kind"] == "verb":
                ordered_add(global_verbs, normalized_value)
            else:
                ordered_add(global_nouns, normalized_value)

        if is_new_pattern:
            section["patterns"][normalized_pattern] = None
            grammar_entries += 1

        section["queries"].append(
            {
                "query": raw_query,
                "normalized_query": normalized_query,
                "pattern": normalized_pattern,
                "semantics": semantics,
                "used_entities": [item["section"] for item in used_occurrences],
            }
        )
        normalized_pattern_sections.setdefault(normalized_pattern, set()).add(section_name)

    grammar_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    intent_map: "OrderedDict[str, dict]" = OrderedDict()
    with grammar_path.open("w", encoding="utf-8") as grammar_file:
        grammar_file.write(
            "# Auto-generated from the MAC-SLU training split.\n"
            "# Intent sections contain utterance patterns.\n"
            "# Slot sections exhaustively enumerate observed noun/verb values.\n"
            "# Regenerate with: ./run_macslu.sh --stage 2 --stop-stage 2\n\n"
        )

        for section_name, section in sections.items():
            if not section["patterns"]:
                continue
            grammar_file.write(f"[{section_name}]\n")
            for pattern in section["patterns"]:
                grammar_file.write(f"{pattern}\n")
            grammar_file.write("\n")
            intent_map[section_name] = {
                "intent_signature": section["intent_signature"],
                "queries": section["queries"],
                "entity_map": section["entity_map"],
                "training_count": section["training_count"],
                "grammar_entries": len(section["patterns"]),
            }

        for slot_section, values in slot_values.items():
            if not values:
                continue
            grammar_file.write(f"[{slot_section}]\n")
            for value in values:
                grammar_file.write(f"{value}\n")
            grammar_file.write("\n")

    map_path.write_text(
        json.dumps(intent_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = {
        "input_rows": len(rows),
        "empty_queries": empty_queries,
        "zero_intent_rows": zero_intent_rows,
        "single_intent_rows": single_intent_rows,
        "multi_intent_rows": multi_intent_rows,
        "intent_sections": len(intent_map),
        "grammar_entries": grammar_entries,
        "slot_sections": len(slot_values),
        "noun_values": len(global_nouns),
        "verb_values": len(global_verbs),
        "ambiguous_patterns": sum(
            1 for names in normalized_pattern_sections.values() if len(names) > 1
        ),
        "dictionary_files": args.dictionary,
        "known_words": len(known_words),
        "jieba_pos_enabled": bool(pseg is not None and not args.disable_pos_lexicon),
        "grammar_file": str(grammar_path),
        "intent_map_file": str(map_path),
    }
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
