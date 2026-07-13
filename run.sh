#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Default configuration
# ============================================================

stage=0
stop_stage=7

profile="zh-cn_pocketsphinx-cmu"
image="voice2json-zh:local"

grammar_file="conf/grammar_zh.ini"

test_text="打开客厅的灯"

test_wav="wavs/coral_gpt-4o-mini-tts_1x_2026-07-12T09_47_27-805Z.wav"
normalized_wav="wavs/test_16k.wav"

ram_repeat=100

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile_dir="${HOME}/.local/share/voice2json/${profile}"
exp_dir="${root_dir}/exp/${profile}"

# ============================================================
# Parse command-line options
# ============================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            stage="$2"
            shift 2
            ;;
        --stop_stage|--stop-stage)
            stop_stage="$2"
            shift 2
            ;;
        --profile)
            profile="$2"
            profile_dir="${HOME}/.local/share/voice2json/${profile}"
            shift 2
            ;;
        --image)
            image="$2"
            shift 2
            ;;
        --grammar-file)
            grammar_file="$2"
            shift 2
            ;;
        --test-text)
            test_text="$2"
            shift 2
            ;;
        --test-wav)
            test_wav="$2"
            shift 2
            ;;
        --normalized-wav)
            normalized_wav="$2"
            shift 2
            ;;
        --ram-repeat)
            ram_repeat="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${exp_dir}"

grammar_path="${root_dir}/${grammar_file}"
test_wav_path="${root_dir}/${test_wav}"
normalized_wav_path="${root_dir}/${normalized_wav}"

v2j() {
    VOICE2JSON_IMAGE="${image}" \
        "${root_dir}/run_voice2json.sh" \
        --profile "${profile}" "$@"
}

pretty_json() {
    if command -v jq >/dev/null 2>&1; then
        jq .
    else
        cat
    fi
}

human_bytes() {
    numfmt --to=iec-i --suffix=B "$1"
}

path_bytes() {
    local path="$1"

    if [[ -e "${path}" ]]; then
        du -sb "${path}" | awk '{print $1}'
    else
        echo 0
    fi
}

sum_path_bytes() {
    local total=0
    local path
    local size

    for path in "$@"; do
        size="$(path_bytes "${path}")"
        total=$((total + size))
    done

    echo "${total}"
}

# ============================================================
# Stage 0: Build patched Docker image
# ============================================================

if (( stage <= 0 && stop_stage >= 0 )); then
    echo "============================================================"
    echo "Stage 0: Build patched Docker image"
    echo "============================================================"

    mkdir -p "${root_dir}/docker_patch"

    profile_definition="${root_dir}/etc/profiles/${profile}.yml"
    patched_definition="${root_dir}/docker_patch/${profile}.yml"

    if [[ -s "${profile_definition}" ]]; then
        cp "${profile_definition}" "${patched_definition}"
    else
        echo "Profile definition is missing from the current checkout."
        echo "Trying the official v2.1 tag."

        git -C "${root_dir}" fetch --tags

        git -C "${root_dir}" show \
            "v2.1:etc/profiles/${profile}.yml" \
            > "${patched_definition}"
    fi

    if [[ ! -s "${patched_definition}" ]]; then
        echo "Failed to obtain profile definition:" >&2
        echo "${patched_definition}" >&2
        exit 1
    fi

    cat > "${root_dir}/docker_patch/Dockerfile" <<DOCKER_EOF
FROM synesthesiam/voice2json:latest

COPY ${profile}.yml \
     /usr/lib/voice2json/etc/profiles/${profile}.yml
DOCKER_EOF

    docker build \
        -t "${image}" \
        "${root_dir}/docker_patch"

    docker run --rm \
        --entrypoint bash \
        "${image}" \
        -lc "test -s /usr/lib/voice2json/etc/profiles/${profile}.yml"

    echo "Docker image is ready: ${image}"
fi

# ============================================================
# Stage 1: Download Mandarin profile
# ============================================================

if (( stage <= 1 && stop_stage >= 1 )); then
    echo "============================================================"
    echo "Stage 1: Download Mandarin profile"
    echo "============================================================"

    VOICE2JSON_IMAGE="${image}" \
        "${root_dir}/run_voice2json.sh" \
        --debug \
        --profile "${profile}" \
        download-profile

    if [[ ! -d "${profile_dir}" ]]; then
        echo "Profile directory was not created:" >&2
        echo "${profile_dir}" >&2
        exit 1
    fi

    echo "Profile directory:"
    echo "${profile_dir}"
fi

# ============================================================
# Stage 2: Load external grammar into profile
# ============================================================

if (( stage <= 2 && stop_stage >= 2 )); then
    echo "============================================================"
    echo "Stage 2: Load external grammar"
    echo "============================================================"

    if [[ ! -s "${grammar_path}" ]]; then
        echo "Grammar file is missing or empty:" >&2
        echo "${grammar_path}" >&2
        exit 1
    fi

    if [[ ! -d "${profile_dir}" ]]; then
        echo "Profile is not downloaded yet:" >&2
        echo "${profile_dir}" >&2
        exit 1
    fi

    if [[ -f "${profile_dir}/sentences.ini" ]] &&
       [[ ! -f "${profile_dir}/sentences.ini.original" ]]; then
        cp "${profile_dir}/sentences.ini" \
           "${profile_dir}/sentences.ini.original"
    fi

    cp "${grammar_path}" \
       "${profile_dir}/sentences.ini"

    echo "Grammar source:"
    echo "${grammar_path}"

    echo
    echo "Grammar destination:"
    echo "${profile_dir}/sentences.ini"

    echo
    echo "Loaded grammar:"
    cat "${profile_dir}/sentences.ini"
fi

# ============================================================
# Stage 3: Compile grammar and train profile
# ============================================================

if (( stage <= 3 && stop_stage >= 3 )); then
    echo "============================================================"
    echo "Stage 3: Train profile"
    echo "============================================================"

    v2j --debug train-profile

    echo "Training artifacts:"
    ls -lh \
        "${profile_dir}/language_model.txt" \
        "${profile_dir}/dictionary.txt" \
        "${profile_dir}/intent.pickle.gz"
fi

# ============================================================
# Stage 4: Text-only intent test
# ============================================================

if (( stage <= 4 && stop_stage >= 4 )); then
    echo "============================================================"
    echo "Stage 4: Text intent test"
    echo "============================================================"

    echo "Input text:"
    echo "${test_text}"

    v2j recognize-intent \
        --text-input \
        "${test_text}" \
        | tee "${exp_dir}/text_intent.json" \
        | pretty_json
fi

# ============================================================
# Stage 5: Normalize test WAV
# ============================================================

if (( stage <= 5 && stop_stage >= 5 )); then
    echo "============================================================"
    echo "Stage 5: Normalize WAV"
    echo "============================================================"

    if [[ ! -f "${test_wav_path}" ]]; then
        echo "Input WAV does not exist:" >&2
        echo "${test_wav_path}" >&2
        exit 1
    fi

    mkdir -p "$(dirname "${normalized_wav_path}")"

    ffmpeg -y \
        -ignore_length 1 \
        -i "${test_wav_path}" \
        -ar 16000 \
        -ac 1 \
        -c:a pcm_s16le \
        "${normalized_wav_path}"

    soxi "${normalized_wav_path}"
fi

# ============================================================
# Stage 6: Speech recognition and intent recognition
# ============================================================

if (( stage <= 6 && stop_stage >= 6 )); then
    echo "============================================================"
    echo "Stage 6: Speech-to-intent test"
    echo "============================================================"

    if [[ ! -f "${normalized_wav_path}" ]]; then
        echo "Normalized WAV does not exist:" >&2
        echo "${normalized_wav_path}" >&2
        echo "Run Stage 5 first." >&2
        exit 1
    fi

    echo "Closed-grammar transcription:"

    v2j transcribe-wav "${normalized_wav}" \
        | tee "${exp_dir}/transcription.json" \
        | pretty_json

    echo
    echo "Speech-to-intent result:"

    cat "${exp_dir}/transcription.json" \
        | v2j recognize-intent \
        | tee "${exp_dir}/speech_intent.json" \
        | pretty_json

    echo
    echo "Open transcription diagnostic:"

    v2j transcribe-wav \
        --open \
        "${normalized_wav}" \
        | tee "${exp_dir}/open_transcription.json" \
        | pretty_json
fi

# ============================================================
# Stage 7: Resource report
#
# 1. Disk Size
# 2. RAM usage
# 3. Parameter count
# ============================================================

if (( stage <= 7 && stop_stage >= 7 )); then
    echo "============================================================"
    echo "Stage 7: Resource report"
    echo "============================================================"

    if [[ ! -d "${profile_dir}" ]]; then
        echo "Profile directory does not exist:" >&2
        echo "${profile_dir}" >&2
        exit 1
    fi

    report_file="${exp_dir}/resource_report.txt"

    full_profile_bytes="$(path_bytes "${profile_dir}")"

    closed_runtime_bytes="$(
        sum_path_bytes \
            "${profile_dir}/acoustic_model" \
            "${profile_dir}/language_model.txt" \
            "${profile_dir}/dictionary.txt" \
            "${profile_dir}/intent.pickle.gz"
    )"

    open_lm_bytes="$(
        path_bytes "${profile_dir}/base_language_model.txt"
    )"

    docker_image_bytes="$(
        docker image inspect "${image}" \
            --format '{{.Size}}'
    )"

    # --------------------------------------------------------
    # Measure peak container RAM
    # --------------------------------------------------------

    if [[ ! -f "${normalized_wav_path}" ]]; then
        echo "Normalized WAV is required for RAM measurement:" >&2
        echo "${normalized_wav_path}" >&2
        echo "Run Stage 5 first." >&2
        exit 1
    fi

    ram_container="voice2json_ram_${$}_${RANDOM}"
    ram_stats_file="$(mktemp)"
    ram_input_file="$(mktemp)"

    cleanup() {
        docker rm -f "${ram_container}" >/dev/null 2>&1 || true
        rm -f "${ram_stats_file}" "${ram_input_file}"
    }

    trap cleanup EXIT

    for _ in $(seq 1 "${ram_repeat}"); do
        echo "${normalized_wav}"
    done > "${ram_input_file}"

    docker run --rm -i \
        --name "${ram_container}" \
        --init \
        -v "${HOME}:${HOME}" \
        -v "${root_dir}:/workspace" \
        -v "/dev/shm:/dev/shm" \
        -w "/workspace" \
        -e "HOME=${HOME}" \
        --user "$(id -u):$(id -g)" \
        "${image}" \
        --profile "${profile}" \
        transcribe-wav \
        --stdin-files \
        < "${ram_input_file}" \
        >/dev/null 2>&1 &

    ram_job_pid=$!

    for _ in $(seq 1 100); do
        if docker inspect "${ram_container}" >/dev/null 2>&1; then
            break
        fi

        sleep 0.05
    done

    while kill -0 "${ram_job_pid}" >/dev/null 2>&1; do
        docker stats \
            --no-stream \
            --format '{{.MemUsage}}' \
            "${ram_container}" \
            2>/dev/null \
            | awk -F' / ' '{print $1}' \
            >> "${ram_stats_file}" || true

        sleep 0.1
    done

    wait "${ram_job_pid}"

    peak_ram_bytes="$(
        python3 - "${ram_stats_file}" <<'PY'
import re
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])

unit_scale = {
    "B": 1,
    "kB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
}

values = []

if stats_path.exists():
    for line in stats_path.read_text().splitlines():
        match = re.search(r"([0-9.]+)\s*([A-Za-z]+)", line)

        if not match:
            continue

        value = float(match.group(1))
        unit = match.group(2)

        if unit in unit_scale:
            values.append(int(value * unit_scale[unit]))

print(max(values) if values else 0)
PY
    )"

    # --------------------------------------------------------
    # Parameter-like storage estimate
    # --------------------------------------------------------

    parameter_report="$(
        python3 - "${profile_dir}" <<'PY'
import sys
from pathlib import Path

profile_dir = Path(sys.argv[1])
acoustic_dir = profile_dir / "acoustic_model"

numeric_files = [
    acoustic_dir / "means",
    acoustic_dir / "variances",
    acoustic_dir / "mixture_weights",
    acoustic_dir / "transition_matrices",
]

total_bytes = 0

for path in numeric_files:
    if path.exists():
        size = path.stat().st_size
        total_bytes += size
        print(
            f"  {path.name:24s}: "
            f"{size / (1024 ** 2):10.3f} MiB"
        )
    else:
        print(f"  {path.name:24s}: MISSING")

float32_equivalent = total_bytes / 4

print()
print("  Neural parameter count: N/A")
print("  Model family: Pocketsphinx GMM-HMM")
print(
    "  Float32-equivalent numeric scalar estimate: "
    f"{float32_equivalent:,.0f}"
)
print(
    "  Numeric acoustic storage: "
    f"{total_bytes / (1024 ** 2):.3f} MiB"
)
print(
    "  Note: this is a storage-derived estimate, "
    "not an exact neural parameter count."
)
PY
    )"

    {
        echo "voice2json Resource Report"
        echo "Profile: ${profile}"
        echo "Image: ${image}"
        echo

        echo "1. Disk Size"
        echo "  Full profile:"
        echo "    $(human_bytes "${full_profile_bytes}")"
        echo
        echo "  Closed-grammar runtime files:"
        echo "    acoustic_model/"
        echo "    language_model.txt"
        echo "    dictionary.txt"
        echo "    intent.pickle.gz"
        echo "    Total: $(human_bytes "${closed_runtime_bytes}")"
        echo
        echo "  Open-transcription language model:"
        echo "    $(human_bytes "${open_lm_bytes}")"
        echo
        echo "  Docker image virtual size:"
        echo "    $(human_bytes "${docker_image_bytes}")"
        echo

        echo "2. RAM usage"
        echo "  Measurement mode:"
        echo "    Closed-grammar transcribe-wav"
        echo "  Repeated utterances:"
        echo "    ${ram_repeat}"
        echo "  Peak inference container RAM:"
        echo "    $(human_bytes "${peak_ram_bytes}")"
        echo

        echo "3. Parameter count"
        echo "${parameter_report}"
    } | tee "${report_file}"

    trap - EXIT
    cleanup

    echo
    echo "Resource report saved to:"
    echo "${report_file}"
fi
