#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run MAC-SLU inference with voice2json.

Modes:
- audio: open/closed Pocketsphinx ASR, followed by fuzzy voice2json intent matching
- oracle_text: use reference query text, followed by fuzzy intent matching

The synthetic voice2json intent is converted back to the complete MAC-SLU
semantics list through an external JSON mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from build_macslu_grammar import load_dictionary_words, normalize_for_voice2json


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


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_jsonl_command(cmd: List[str], input_lines: List[str]) -> List[dict]:
    payload = "".join(line.rstrip("\n") + "\n" for line in input_lines)
    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}"
        )

    outputs: List[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            outputs.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[WARN] Ignoring non-JSON stdout: {line}", file=sys.stderr)

    return outputs


def recognize_texts(
    texts: List[str],
    voice2json_cmd: List[str],
    batch_size: int,
) -> List[dict]:
    results: List[dict] = [{} for _ in texts]
    valid_indices = [i for i, text in enumerate(texts) if text.strip()]

    for index_chunk in chunked(valid_indices, batch_size):
        input_lines = [texts[i] for i in index_chunk]
        cmd = voice2json_cmd + [
            "recognize-intent",
            "--text-input",
        ]
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
        cmd = voice2json_cmd + [
            "transcribe-wav",
            "--stdin-files",
        ]
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
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--intent-map", required=True)
    p.add_argument("--run-voice2json", default="")
    p.add_argument("--image", default="voice2json-zh:local")
    p.add_argument("--mount", action="append", default=[])
    p.add_argument("--profile", default="zh-cn_pocketsphinx-cmu")
    p.add_argument(
        "--decode-mode",
        choices=["audio", "oracle_text"],
        default="audio",
    )
    p.add_argument(
        "--asr-mode",
        choices=["open", "closed"],
        default="open",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--dictionary",
        action="append",
        required=True,
        help=(
            "Pronunciation dictionary to use as a known-word allowlist during "
            "text normalization. May be passed multiple times."
        ),
    )
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    intent_map = json.loads(Path(args.intent_map).read_text(encoding="utf-8"))
    rows = load_jsonl(input_path)
    known_words = load_dictionary_words(Path(p) for p in args.dictionary)
    if not known_words:
        raise ValueError(
            "No known words were loaded from --dictionary; cannot normalize "
            "MAC-SLU inputs with the grammar's OOV filter."
        )
    voice2json_cmd = get_voice2json_cmd(args)

    if args.decode_mode == "audio":
        audio_paths = [str(row.get("audio", "")) for row in rows]
        missing = [p for p in audio_paths if not p or not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} audio path(s) are missing. First: {missing[0]}"
            )
        asr_outputs = transcribe_audio(
            audio_paths,
            voice2json_cmd,
            args.asr_mode,
            args.batch_size,
        )
        pred_queries = [str(x.get("text", "")) for x in asr_outputs]
    else:
        asr_outputs = [{} for _ in rows]
        pred_queries = [str(row.get("query", "")) for row in rows]

    normalized_queries = [
        normalize_for_voice2json(
            query,
            known_words=known_words,
        )
        for query in pred_queries
    ]
    nlu_outputs = recognize_texts(
        normalized_queries,
        voice2json_cmd,
        args.batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row, pred_query, norm_query, asr_result, nlu_result in zip(
            rows,
            pred_queries,
            normalized_queries,
            asr_outputs,
            nlu_outputs,
        ):
            intent_name = (
                nlu_result.get("intent", {}).get("name", "")
                if isinstance(nlu_result, dict)
                else ""
            )
            map_entry = intent_map.get(intent_name, {})
            pred_semantics = map_entry.get("query_semantics", {}).get(norm_query, [])
            matched_queries = map_entry.get("queries", [])
            matched_train_query = ""
            if matched_queries:
                for matched_query in matched_queries:
                    if matched_query.get("normalized_query", "") == norm_query:
                        matched_train_query = matched_query.get("query", "")
                        break
                if not matched_train_query:
                    matched_train_query = matched_queries[0].get("query", "")
            if map_entry:
                matched += 1

            output = {
                "text_id": row.get("text_id", row.get("id", "")),
                "query": row.get("query", ""),
                "pred_query": pred_query,
                "pred_semantics": pred_semantics,
                # Compatibility with the official MAC-SLU metrics.py
                "semantics": pred_semantics,
                "voice2json_intent": intent_name,
                "voice2json_confidence": (
                    nlu_result.get("intent", {}).get("confidence", 0.0)
                    if isinstance(nlu_result, dict)
                    else 0.0
                ),
                "normalized_pred_query": norm_query,
                "matched_train_query": matched_train_query,
                "asr_likelihood": (
                    asr_result.get("likelihood", 0.0)
                    if isinstance(asr_result, dict)
                    else 0.0
                ),
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "num_examples": len(rows),
                "mapped_intents": matched,
                "unmapped_intents": len(rows) - matched,
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
