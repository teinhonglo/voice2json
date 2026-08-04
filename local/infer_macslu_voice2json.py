#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run MAC-SLU inference with voice2json.

A synthetic voice2json intent represents an ordered MAC-SLU intent sequence.
Recognized grammar entities are mapped back to MAC-SLU frame/slot positions.
The closest training template supplies implicit slots and any explicit values
that are not recoverable from the recognized entity list.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from build_macslu_grammar import load_dictionary_words, normalize_for_voice2json


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


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_jsonl_command(cmd: List[str], input_lines: List[str]) -> List[dict]:
    payload = "".join(line.rstrip("\n") + "\n" for line in input_lines)
    process = subprocess.run(
        cmd,
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        print(process.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {' '.join(cmd)}"
        )

    outputs: List[dict] = []
    for line in process.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            outputs.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[WARN] Ignoring non-JSON stdout: {line}", file=sys.stderr)
    return outputs


def recognize_texts(
    texts: List[str], voice2json_cmd: List[str], batch_size: int
) -> List[dict]:
    results: List[dict] = [{} for _ in texts]
    valid_indices = [index for index, text in enumerate(texts) if text.strip()]
    for index_chunk in chunked(valid_indices, batch_size):
        input_lines = [texts[index] for index in index_chunk]
        cmd = voice2json_cmd + ["recognize-intent", "--text-input"]
        chunk_outputs = run_jsonl_command(cmd, input_lines)
        if len(chunk_outputs) != len(index_chunk):
            raise RuntimeError(
                "recognize-intent line-count mismatch: "
                f"expected={len(index_chunk)} got={len(chunk_outputs)}"
            )
        for original_index, output in zip(index_chunk, chunk_outputs):
            results[original_index] = output
    return results


def transcribe_audio(
    audio_paths: List[str],
    voice2json_cmd: List[str],
    asr_mode: str,
    batch_size: int,
) -> List[dict]:
    results: List[dict] = []
    for path_chunk in chunked(audio_paths, batch_size):
        cmd = voice2json_cmd + ["transcribe-wav", "--stdin-files"]
        if asr_mode == "open":
            cmd.append("--open")
        chunk_outputs = run_jsonl_command(cmd, list(path_chunk))
        if len(chunk_outputs) != len(path_chunk):
            raise RuntimeError(
                "transcribe-wav line-count mismatch: "
                f"expected={len(path_chunk)} got={len(chunk_outputs)}"
            )
        results.extend(chunk_outputs)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--intent-map", required=True)
    parser.add_argument("--run-voice2json", default="")
    parser.add_argument("--image", default="voice2json-zh:local")
    parser.add_argument("--mount", action="append", default=[])
    parser.add_argument("--profile", default="zh-cn_pocketsphinx-cmu")
    parser.add_argument(
        "--decode-mode", choices=["audio", "oracle_text"], default="audio"
    )
    parser.add_argument("--asr-mode", choices=["open", "closed"], default="open")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--dictionary",
        action="append",
        required=True,
        help="Pronunciation dictionary allowlist. May be supplied multiple times.",
    )
    return parser.parse_args()


def get_voice2json_cmd(args: argparse.Namespace) -> List[str]:
    if args.run_voice2json:
        return [args.run_voice2json, "--profile", args.profile]

    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--init",
        "-v",
        f"{Path.home()}:{Path.home()}",
        "-v",
        "/dev/shm:/dev/shm",
        "-w",
        str(Path.cwd()),
        "-e",
        f"HOME={Path.home()}",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    for mount_path in args.mount:
        path = Path(mount_path)
        if path.is_dir():
            cmd.extend(["-v", f"{path}:{path}"])
    cmd.extend([args.image, "--profile", args.profile])
    return cmd


def intent_name(nlu_result: dict) -> str:
    intent = nlu_result.get("intent", {}) if isinstance(nlu_result, dict) else {}
    return str(intent.get("name", "")) if isinstance(intent, dict) else ""


def entity_name(entity: dict) -> str:
    for key in ("entity", "name", "slot"):
        value = entity.get(key)
        if value:
            return str(value)
    return ""


def entity_value(entity: dict) -> str:
    value: Any = entity.get("value", entity.get("raw_value", ""))
    if isinstance(value, dict):
        value = value.get("value", value.get("text", ""))
    if isinstance(value, list):
        value = "".join(str(item) for item in value)
    text = str(value or "").strip()
    # The Mandarin profile commonly returns character-spaced entity values.
    if text and all(len(token) == 1 for token in text.split()):
        text = "".join(text.split())
    return text


def base_frames(map_entry: dict) -> List[dict]:
    frames = map_entry.get("intent_signature", [])
    if not isinstance(frames, list):
        return []
    output: List[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        output.append(
            {
                "domain": str(frame.get("domain") or ""),
                "intent": str(frame.get("intent") or ""),
                "slots": {},
                "implicit_slots": {},
            }
        )
    return output


def nearest_template(
    map_entry: dict, normalized_query: str
) -> Tuple[dict, float]:
    queries = map_entry.get("queries", [])
    if not isinstance(queries, list) or not queries:
        return {}, 0.0

    for query in queries:
        if (
            isinstance(query, dict)
            and query.get("normalized_query", "") == normalized_query
        ):
            return query, 1.0

    best: dict = {}
    best_score = -1.0
    compact_target = normalized_query.replace(" ", "")
    for query in queries:
        if not isinstance(query, dict):
            continue
        candidate = str(query.get("normalized_query", ""))
        score = SequenceMatcher(
            None, compact_target, candidate.replace(" ", "")
        ).ratio()
        if score > best_score:
            best = query
            best_score = score
    return best, max(best_score, 0.0)


def reconstruct_semantics(
    map_entry: dict,
    normalized_query: str,
    nlu_result: dict,
) -> Tuple[List[dict], str, float]:
    """Reconstruct MAC-SLU semantics from a recognized synthetic intent."""
    template, similarity = nearest_template(map_entry, normalized_query)
    template_semantics = template.get("semantics", []) if template else []
    if isinstance(template_semantics, list):
        semantics = copy.deepcopy(template_semantics)
    else:
        semantics = []
    if not semantics:
        semantics = base_frames(map_entry)

    entity_map = map_entry.get("entity_map", {})
    if not isinstance(entity_map, dict):
        entity_map = {}
    entities = nlu_result.get("entities", []) if isinstance(nlu_result, dict) else []
    if not isinstance(entities, list):
        entities = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        grammar_entity = entity_name(entity)
        metadata = entity_map.get(grammar_entity)
        if not isinstance(metadata, dict):
            continue
        frame_index = int(metadata.get("frame_index", -1))
        slot_name = str(metadata.get("slot_name") or "")
        value = entity_value(entity)
        if frame_index < 0 or frame_index >= len(semantics) or not slot_name or not value:
            continue
        frame = semantics[frame_index]
        if not isinstance(frame, dict):
            continue
        slots = frame.setdefault("slots", {})
        if isinstance(slots, dict):
            slots[slot_name] = value

    matched_query = str(template.get("query", "")) if template else ""
    return semantics, matched_query, similarity


def legacy_semantics(map_entry: dict, normalized_query: str) -> List[dict]:
    query_semantics = map_entry.get("query_semantics", {})
    if not isinstance(query_semantics, dict):
        return []
    value = query_semantics.get(normalized_query, [])
    return value if isinstance(value, list) else []


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    intent_map = json.loads(Path(args.intent_map).read_text(encoding="utf-8"))
    rows = load_jsonl(input_path)
    known_words: Set[str] = load_dictionary_words(Path(path) for path in args.dictionary)
    if not known_words:
        raise ValueError("No known words were loaded from --dictionary")
    voice2json_cmd = get_voice2json_cmd(args)

    if args.decode_mode == "audio":
        audio_paths = [str(row.get("audio", "")) for row in rows]
        missing = [path for path in audio_paths if not path or not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} audio path(s) are missing. First: {missing[0]}"
            )
        asr_outputs = transcribe_audio(
            audio_paths, voice2json_cmd, args.asr_mode, args.batch_size
        )
        pred_queries = [str(result.get("text", "")) for result in asr_outputs]
    else:
        asr_outputs = [{} for _ in rows]
        pred_queries = [str(row.get("query", "")) for row in rows]

    normalized_queries = [
        normalize_for_voice2json(query, known_words) for query in pred_queries
    ]
    nlu_outputs = recognize_texts(
        normalized_queries, voice2json_cmd, args.batch_size
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for row, pred_query, norm_query, asr_result, nlu_result in zip(
            rows, pred_queries, normalized_queries, asr_outputs, nlu_outputs
        ):
            recognized_intent = intent_name(nlu_result)
            map_entry = intent_map.get(recognized_intent, {})
            matched_train_query = ""
            template_similarity = 0.0
            if isinstance(map_entry, dict) and "intent_signature" in map_entry:
                pred_semantics, matched_train_query, template_similarity = (
                    reconstruct_semantics(map_entry, norm_query, nlu_result)
                )
            elif isinstance(map_entry, dict):
                pred_semantics = legacy_semantics(map_entry, norm_query)
            else:
                pred_semantics = []
            if map_entry:
                mapped += 1

            intent_result = (
                nlu_result.get("intent", {}) if isinstance(nlu_result, dict) else {}
            )
            output = {
                "text_id": row.get("text_id", row.get("id", "")),
                "query": row.get("query", ""),
                "pred_query": pred_query,
                "pred_semantics": pred_semantics,
                "semantics": pred_semantics,
                "voice2json_intent": recognized_intent,
                "voice2json_confidence": (
                    intent_result.get("confidence", 0.0)
                    if isinstance(intent_result, dict)
                    else 0.0
                ),
                "normalized_pred_query": norm_query,
                "matched_train_query": matched_train_query,
                "template_similarity": template_similarity,
                "recognized_entities": (
                    nlu_result.get("entities", [])
                    if isinstance(nlu_result, dict)
                    else []
                ),
                "asr_likelihood": (
                    asr_result.get("likelihood", 0.0)
                    if isinstance(asr_result, dict)
                    else 0.0
                ),
            }
            output_file.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "num_examples": len(rows),
                "mapped_intents": mapped,
                "unmapped_intents": len(rows) - mapped,
                "decode_mode": args.decode_mode,
                "asr_mode": args.asr_mode,
                "output_jsonl": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
