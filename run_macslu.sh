#!/usr/bin/env bash
# voice2json baseline for MAC-SLU
#
# Stages 0-2 are setup/training stages and normally run only once.
# Therefore the default stage is 3 (inference), following the convention
# used in Qwen3-SLU/run_macslu.sh.

set -Eeuo pipefail

help_message=$(cat <<'EOF'
Usage: ./run_macslu.sh [options]

Main options:
  --stage N
  --stop-stage N
  --decode-mode audio|oracle_text
  --asr-mode open|closed
  --test-sets "test"
  --gpuid N                 # accepted for interface consistency; unused
  --qwen3-slu-root PATH
  --max-grammar-entries N   # 0 means all training queries

Stages:
  -1  Build patched Docker image
   0  Download profile and prepare MAC-SLU JSONL
   1  Build grammar and intent map
   2  Train voice2json profile
   3  Run inference
   4  Evaluate predictions
   5  Report resources
EOF
)

# -----------------------------
# Data configuration
# -----------------------------
repo_id="Gatsby1984/MAC_SLU"
data_root="data/macslu"
download_dir="${data_root}/raw"
extract_root="${data_root}/audio"
json_root="data-json/macslu"

# Reuse the user's established MAC-SLU preparation and evaluation scripts.
qwen3_slu_root="../Qwen3-SLU"
prepare_py=""
metrics_py=""

# -----------------------------
# voice2json configuration
# -----------------------------
profile="zh-cn_pocketsphinx-cmu"
image="voice2json-zh:local"
grammar_file="conf/macslu_train_grammar.ini"
intent_map_file="conf/macslu_intent_map.json"
grammar_stats_file="conf/macslu_grammar_stats.json"
max_grammar_entries=0

# -----------------------------
# Inference/evaluation config
# -----------------------------
decode_mode="audio"       # audio | oracle_text
asr_mode="open"           # open is recommended for unseen MAC-SLU queries
test_sets="test"
batch_size=256
exp_root="exp/macslu/voice2json"

# Kept for command-line compatibility with the Qwen3-SLU scripts.
# Pocketsphinx does not use CUDA.
gpuid=0
nj=1

# -----------------------------
# Resource reporting
# -----------------------------
ram_repeat=50

# -----------------------------
# Stage control
# -----------------------------
# Stages -1 through 2 are setup/training. Default starts at inference.
stage=3
stop_stage=4

. ./local/parse_options.sh
. ./path.sh

trap 'exit_code=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line ${LINENO}: ${BASH_COMMAND}" >&2; exit "${exit_code}"' ERR

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
qwen3_slu_root="$(realpath "${qwen3_slu_root}")"

if [ -z "${prepare_py}" ]; then
    prepare_py="${qwen3_slu_root}/local/prepare_macslu_jsonl.py"
fi
if [ -z "${metrics_py}" ]; then
    metrics_py="${qwen3_slu_root}/local/metrics.py"
fi

profile_dir="${HOME}/.local/share/voice2json/${profile}"
exp_root="${exp_root}/${decode_mode}_${asr_mode}"
run_voice2json="${root_dir}/run_voice2json.sh"
export VOICE2JSON_IMAGE="${image}"
export VOICE2JSON_MOUNTS="${root_dir} ${qwen3_slu_root}"

mkdir -p "${exp_root}"

require_file() {
    local path="$1"
    local message="${2:-missing required file}"

    if [ ! -s "${path}" ]; then
        echo "[ERROR] ${message}: ${path}" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    local message="${2:-missing required directory}"

    if [ ! -d "${path}" ]; then
        echo "[ERROR] ${message}: ${path}" >&2
        exit 1
    fi
}

v2j() {
    VOICE2JSON_IMAGE="${image}" \
        "${run_voice2json}" \
        --profile "${profile}" "$@"
}

profile_dictionary_args=()
collect_profile_dictionaries() {
    local dict_path

    profile_dictionary_args=()
    for dict_path in \
        "${profile_dir}/base_dictionary.txt" \
        "${profile_dir}/custom_words.txt"; do
        if [ -s "${dict_path}" ]; then
            profile_dictionary_args+=(--dictionary "${dict_path}")
        fi
    done

    if [ "${#profile_dictionary_args[@]}" -eq 0 ]; then
        echo "[ERROR] no pronunciation dictionaries found in profile:" >&2
        echo "${profile_dir}" >&2
        echo "Run Stage 0 first so base_dictionary.txt is available." >&2
        exit 1
    fi
}

show_metrics_files() {
    local test_set
    local metrics_file

    for test_set in ${test_sets}; do
        metrics_file="${exp_root}/${test_set}/metrics.txt"
        if [ -s "${metrics_file}" ]; then
            echo
            echo "========== ${metrics_file} =========="
            cat "${metrics_file}"
        fi
    done
}

# ============================================================
# Stage -1: Build patched Docker image
# ============================================================
if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
    echo "Stage -1: Build patched voice2json Docker image"

    mkdir -p docker_patch

    profile_definition="etc/profiles/${profile}.yml"
    patched_definition="docker_patch/${profile}.yml"

    if [ -s "${profile_definition}" ]; then
        cp "${profile_definition}" "${patched_definition}"
    else
        git fetch --tags
        git show "v2.1:etc/profiles/${profile}.yml" \
            > "${patched_definition}"
    fi

    require_file "${patched_definition}" "failed to obtain ${profile}.yml"

    cat > docker_patch/Dockerfile <<DOCKER_EOF
FROM synesthesiam/voice2json:latest
COPY ${profile}.yml /usr/lib/voice2json/etc/profiles/${profile}.yml
DOCKER_EOF

    docker build -t "${image}" docker_patch
fi

# ============================================================
# Stage 0: Download profile and prepare MAC-SLU JSONL
# ============================================================
if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Download profile and prepare MAC-SLU"

    require_file "${prepare_py}" "prepare script not found"

    if ! docker image inspect "${image}" >/dev/null 2>&1; then
        echo "[ERROR] Docker image not found: ${image}" >&2
        echo "Run Stage -1 first to build the patched image." >&2
        exit 1
    fi

    if [ ! -d "${profile_dir}/acoustic_model" ] ||
       [ ! -s "${profile_dir}/base_dictionary.txt" ]; then
        VOICE2JSON_IMAGE="${image}" \
            "${run_voice2json}" \
            --debug \
            --profile "${profile}" \
            download-profile
    else
        echo "[INFO] profile already exists: ${profile_dir}"
    fi

    prep_cmd=(
        python "${prepare_py}"
        --repo-id "${repo_id}"
        --download-dir "${download_dir}"
        --extract-root "${extract_root}"
        --jsonl-root "${json_root}"
        --splits train dev test
    )
    "${prep_cmd[@]}"
fi

# ============================================================
# Stage 1: Build external grammar and multi-intent mapping
# ============================================================
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Generate MAC-SLU grammar from training queries"

    require_file "${json_root}/train.jsonl" "MAC-SLU train JSONL not found; run Stage 0 first"
    require_dir "${profile_dir}" "profile directory not found; run Stage 0 first"
    collect_profile_dictionaries

    grammar_cmd=(
        python local/build_macslu_grammar.py
        --train-jsonl "${json_root}/train.jsonl"
        --grammar-out "${grammar_file}"
        --intent-map-out "${intent_map_file}"
        --stats-out "${grammar_stats_file}"
        --max-entries "${max_grammar_entries}"
    )
    grammar_cmd+=("${profile_dictionary_args[@]}")
    "${grammar_cmd[@]}"
fi

# ============================================================
# Stage 2: Load grammar and compile voice2json profile
# ============================================================
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Load grammar and train voice2json profile"

    require_file "${grammar_file}" "grammar file missing; run Stage 1 first"
    require_file "${intent_map_file}" "intent map missing; run Stage 1 first"
    require_dir "${profile_dir}" "profile directory not found; run Stage 0 first"

    if [ -f "${profile_dir}/sentences.ini" ] &&
       [ ! -f "${profile_dir}/sentences.ini.original" ]; then
        cp "${profile_dir}/sentences.ini" \
           "${profile_dir}/sentences.ini.original"
    fi

    cp "${grammar_file}" "${profile_dir}/sentences.ini"

    rm -f \
        "${profile_dir}/language_model.txt" \
        "${profile_dir}/language_model.fst" \
        "${profile_dir}/intent.fst" \
        "${profile_dir}/intent.pickle.gz" \
        "${profile_dir}/dictionary.txt" \
        "${profile_dir}/unknown_words.txt"

    v2j --debug train-profile
fi

# ============================================================
# Stage 3: Inference
# ============================================================
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Inference on MAC-SLU"

    for test_set in ${test_sets}; do
        test_jsonl="${json_root}/${test_set}.jsonl"
        output_dir="${exp_root}/${test_set}"
        pred_file="${output_dir}/predictions.jsonl"

        require_file "${test_jsonl}" "MAC-SLU ${test_set} JSONL not found; run Stage 0 first"
        require_file "${intent_map_file}" "intent map missing; run Stage 1 first"
        require_dir "${profile_dir}" "profile directory not found; run Stage 0 first"
        collect_profile_dictionaries

        mkdir -p "${output_dir}"

        infer_cmd=(
            python local/infer_macslu_voice2json.py
            --input-jsonl "${test_jsonl}"
            --output-jsonl "${pred_file}"
            --intent-map "${intent_map_file}"
            --run-voice2json "${run_voice2json}"
            --profile "${profile}"
            --decode-mode "${decode_mode}"
            --asr-mode "${asr_mode}"
            --batch-size "${batch_size}"
        )
        infer_cmd+=("${profile_dictionary_args[@]}")
        "${infer_cmd[@]}"
    done
fi

# ============================================================
# Stage 4: Evaluation with the established Qwen3-SLU metrics
# ============================================================
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Evaluate MAC-SLU predictions"

    require_file "${metrics_py}" "metrics script not found"

    for test_set in ${test_sets}; do
        pred_file="${exp_root}/${test_set}/predictions.jsonl"
        gt_file="${json_root}/${test_set}.jsonl"
        output_dir="${exp_root}/${test_set}"

        require_file "${gt_file}" "MAC-SLU ${test_set} JSONL not found; run Stage 0 first"

        if [ ! -f "${pred_file}" ]; then
            echo "[WARNING] prediction file not found: ${pred_file}"
            continue
        fi

        python "${metrics_py}" \
            --output_dir "${output_dir}" \
            "${pred_file}" \
            "${gt_file}" \
            | tee "${output_dir}/metrics.txt"
    done
fi

# ============================================================
# Stage 5: Resource report
#   1. Disk Size
#   2. RAM usage
#   3. Parameter count
# ============================================================
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "Stage 5: Resource report"

    test_jsonl="${json_root}/test.jsonl"
    output_dir="${exp_root}/resource"
    mkdir -p "${output_dir}"

    require_file "${test_jsonl}" "MAC-SLU test JSONL not found; run Stage 0 first"

    first_audio=$(
        python local/model_resource_utils.py \
            first-audio \
            --jsonl "${test_jsonl}"
    )

    if [ -z "${first_audio}" ] || [ ! -f "${first_audio}" ]; then
        echo "[ERROR] could not resolve a test audio file" >&2
        exit 1
    fi

    ram_container="voice2json_ram_${$}_${RANDOM}"
    ram_stats_file="$(mktemp)"
    ram_manifest="$(mktemp)"

    cleanup() {
        docker rm -f "${ram_container}" >/dev/null 2>&1 || true
        rm -f "${ram_stats_file}" "${ram_manifest}"
    }
    trap cleanup EXIT

    for _ in $(seq 1 "${ram_repeat}"); do
        echo "${first_audio}"
    done > "${ram_manifest}"

    docker run --rm -i \
        --name "${ram_container}" \
        --init \
        -v "${HOME}:${HOME}" \
        -v "${root_dir}:${root_dir}" \
        -v "${qwen3_slu_root}:${qwen3_slu_root}:ro" \
        -v "/dev/shm:/dev/shm" \
        -w "${root_dir}" \
        -e "HOME=${HOME}" \
        --user "$(id -u):$(id -g)" \
        "${image}" \
        --profile "${profile}" \
        transcribe-wav \
        --stdin-files \
        --open \
        < "${ram_manifest}" \
        >/dev/null 2>&1 &

    ram_pid=$!

    for _ in $(seq 1 100); do
        if docker inspect "${ram_container}" >/dev/null 2>&1; then
            break
        fi
        sleep 0.05
    done

    while kill -0 "${ram_pid}" >/dev/null 2>&1; do
        docker stats \
            --no-stream \
            --format '{{.MemUsage}}' \
            "${ram_container}" \
            2>/dev/null \
            | awk -F' / ' '{print $1}' \
            >> "${ram_stats_file}" || true
        sleep 0.1
    done

    wait "${ram_pid}"

    peak_ram_bytes=$(
        python local/model_resource_utils.py \
            peak-ram \
            --stats-file "${ram_stats_file}"
    )

    python local/profile_resource_report.py \
        --profile-dir "${profile_dir}" \
        --image "${image}" \
        --peak-ram-bytes "${peak_ram_bytes}" \
        --output-txt "${output_dir}/resource_report.txt" \
        --output-json "${output_dir}/resource_report.json"

    trap - EXIT
    cleanup
fi

if [ "${stop_stage}" -ge 4 ]; then
    show_metrics_files
fi
