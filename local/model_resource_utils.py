#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Small helpers used by run_macslu.sh resource reporting."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def first_audio(jsonl_path: Path) -> str:
    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = str(row.get("audio", ""))
            if audio:
                return audio

    return ""


def parse_peak_ram(stats_path: Path) -> int:
    scale = {
        "B": 1,
        "kB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    values = []
    for line in stats_path.read_text(encoding="utf-8").splitlines():
        match = re.search(r"([0-9.]+)\s*([A-Za-z]+)", line)
        if match and match.group(2) in scale:
            values.append(int(float(match.group(1)) * scale[match.group(2)]))

    return max(values) if values else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    first_audio_parser = subparsers.add_parser("first-audio")
    first_audio_parser.add_argument("--jsonl", required=True)

    peak_ram_parser = subparsers.add_parser("peak-ram")
    peak_ram_parser.add_argument("--stats-file", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "first-audio":
        print(first_audio(Path(args.jsonl)))
    elif args.command == "peak-ram":
        print(parse_peak_ram(Path(args.stats_file)))


if __name__ == "__main__":
    main()
