#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable


def recursive_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--profile-dir", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--peak-ram-bytes", type=int, default=0)
    p.add_argument("--output-txt", required=True)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profile = Path(args.profile_dir)
    acoustic = profile / "acoustic_model"

    closed_paths = [
        acoustic,
        profile / "language_model.txt",
        profile / "dictionary.txt",
        profile / "intent.pickle.gz",
    ]
    numeric_files = [
        acoustic / "means",
        acoustic / "variances",
        acoustic / "mixture_weights",
        acoustic / "transition_matrices",
    ]

    full_profile = recursive_size(profile)
    closed_runtime = sum(recursive_size(p) for p in closed_paths)
    open_lm = recursive_size(profile / "base_language_model.txt")
    numeric_storage = sum(recursive_size(p) for p in numeric_files)
    float32_equivalent = numeric_storage / 4.0

    image_size = int(
        subprocess.check_output(
            [
                "docker",
                "image",
                "inspect",
                args.image,
                "--format",
                "{{.Size}}",
            ],
            text=True,
        ).strip()
    )

    report = {
        "disk_size": {
            "full_profile_bytes": full_profile,
            "closed_runtime_bytes": closed_runtime,
            "open_language_model_bytes": open_lm,
            "docker_image_virtual_bytes": image_size,
        },
        "ram_usage": {
            "peak_inference_container_bytes": args.peak_ram_bytes,
        },
        "parameter_count": {
            "model_family": "Pocketsphinx GMM-HMM",
            "neural_parameter_count": None,
            "numeric_acoustic_storage_bytes": numeric_storage,
            "float32_equivalent_scalar_estimate": float32_equivalent,
            "warning": (
                "Storage-derived scalar estimate only; this is not an exact "
                "neural parameter count."
            ),
        },
    }

    lines = [
        "voice2json Resource Report",
        "",
        "1. Disk Size",
        f"  Full profile: {human_bytes(full_profile)}",
        f"  Closed-grammar runtime files: {human_bytes(closed_runtime)}",
        f"  Open-transcription language model: {human_bytes(open_lm)}",
        f"  Docker image virtual size: {human_bytes(image_size)}",
        "",
        "2. RAM usage",
        f"  Peak inference container RAM: {human_bytes(args.peak_ram_bytes)}",
        "",
        "3. Parameter count",
        "  Model family: Pocketsphinx GMM-HMM",
        "  Neural parameter count: N/A",
        f"  Numeric acoustic storage: {human_bytes(numeric_storage)}",
        (
            "  Float32-equivalent numeric scalar estimate: "
            f"{float32_equivalent:,.0f}"
        ),
        "  Note: storage-derived estimate, not an exact neural parameter count.",
    ]

    Path(args.output_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.output_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
