#!/usr/bin/env bash
set -euo pipefail

# voice2json baseline for MAC-SLU.
# The stage layout intentionally mirrors run.sh wherever the corpus workflow
# allows it: image -> profile/data -> grammar -> train -> text test ->
# diagnostics -> inference -> evaluation/resources.

# ============================================================
# Default configuration
# ============================================================
stage=0
stop_stage=7

profile="zh-cn_pocketsphinx-cmu"
image="voice2json-zh:local"
repo_id="Gatsby1984/MAC_SLU"
qwen3_slu_root="../Qwen3-SLU"
prepare_py=""
metrics_py=""

data_root="data/macslu"
download_dir="${data_root}/raw"
extract_root="${data_root}/audio"
json_root="data-json/macslu"

grammar_file="conf/macslu_train_grammar.ini"
intent_map_file="conf/macslu_intent_map.json"
grammar_stats_file="conf/macslu_grammar_stats.json"
max_grammar_entries=0
disable_pos_lexicon=false

decode_mode="audio"       # audio | oracle_text
asr_mode="closed"          # open | closed
test_sets="test"
test_text=""
batch_size=256
exp_root="exp/macslu/voice2json"
ram_repeat=50
skip_resource_report=false

# Accepted for interface compatibility with Qwen3-SLU. Pocketsphinx is CPU-only.
gpuid=0
nj=1

# ============================================================
# Parse command-line options (same style as run.sh)
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) stage="$2"; shift 2 ;;
        --stop_stage|--stop-stage) stop_stage="$2"; shift 2 ;;
        --profile) profile="$2"; shift 2 ;;
        --image) image="$2"; shift 2 ;;
        --repo-id) repo_id="$2"; shift 2 ;;
        --qwen3-slu-root) qwen3_slu_root="$2"; shift 2 ;;
        --prepare-py) prepare_py="$2"; shift 2 ;;
        --metrics-py) metrics_py="$2"; shift 2 ;;
        --data-root) data_root="$2"; shift 2 ;;
        --download-dir) download_dir="$2"; shift 2 ;;
        --extract-root) extract_root="$2"; shift 2 ;;
        --json-root) json_root="$2"; shift 2 ;;
        --grammar-file) grammar_file="$2"; shift 2 ;;
        --intent-map-file) intent_map_file="$2"; shift 2 ;;
        --grammar-stats-file) grammar_stats_file="$2"; shift 2 ;;
        --max-grammar-entries) max_grammar_entries="$2"; shift 2 ;;
        --disable-pos-lexicon) disable_pos_lexicon=true; shift ;;
        --decode-mode) decode_mode="$2"; shift 2 ;;
        --asr-mode) asr_mode="$2"; shift 2 ;;
        --test-sets) test_sets="$2"; shift 2 ;;
        --test-text) test_text="$2"; shift 2 ;;
        --batch-size) batch_size="$2"; shift 2 ;;
        --exp-root) exp_root="$2"; shift 2 ;;
        --ram-repeat) ram_repeat="$2"; shift 2 ;;
        --skip-resource-report) skip_resource_report=true; shift ;;
        --gpuid) gpuid="$2"; shift 2 ;;
        --nj) nj="$2"; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: ./run_macslu.sh [options]

Stages:
  0  Build patched voice2json Docker image
  1  Download profile and prepare MAC-SLU JSONL
  2  Generate and load MAC-SLU grammar
  3  Compile grammar and train profile
  4  Text-only intent smoke test
  5  Grammar and dataset diagnostics
  6  MAC-SLU inference
  7  Evaluation and resource report

Important options:
  --stage N --stop-stage N
  --decode-mode audio|oracle_text
  --asr-mode open|closed
  --test-sets "dev test"
  --qwen3-slu-root PATH
  --max-grammar-entries N     0 keeps every unique training pattern
  --disable-pos-lexicon       use semantic slots only
  --skip-resource-report
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if (( stage < 0 || stop_stage > 7 || stage > stop_stage )); then
    echo "Invalid stage range: stage=${stage}, stop_stage=${stop_stage}; expected 0..7" >&2
    exit 1
fi
if [[ "${decode_mode}" != "audio" && "${decode_mode}" != "oracle_text" ]]; then
    echo "Invalid --decode-mode: ${decode_mode}" >&2
    exit 1
fi
if [[ "${asr_mode}" != "open" && "${asr_mode}" != "closed" ]]; then
    echo "Invalid --asr-mode: ${asr_mode}" >&2
    exit 1
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
qwen3_slu_root="$(cd "${root_dir}" && realpath -m "${qwen3_slu_root}")"
profile_dir="${HOME}/.local/share/voice2json/${profile}"
grammar_path="${root_dir}/${grammar_file}"
intent_map_path="${root_dir}/${intent_map_file}"
grammar_stats_path="${root_dir}/${grammar_stats_file}"
exp_dir="${root_dir}/${exp_root}/${decode_mode}_${asr_mode}"
run_voice2json="${root_dir}/run_voice2json.sh"

if [[ -z "${prepare_py}" ]]; then
    prepare_py="${qwen3_slu_root}/local/prepare_macslu_jsonl.py"
fi
if [[ -z "${metrics_py}" ]]; then
    metrics_py="${qwen3_slu_root}/local/metrics.py"
fi

mkdir -p "${exp_dir}"
export VOICE2JSON_IMAGE="${image}"
export VOICE2JSON_MOUNTS="${root_dir} ${qwen3_slu_root}"

v2j() {
    VOICE2JSON_IMAGE="${image}" \
        VOICE2JSON_MOUNTS="${VOICE2JSON_MOUNTS}" \
        "${run_voice2json}" --profile "${profile}" "$@"
}

require_file() {
    local path="$1"
    local message="${2:-Required file is missing}"
    if [[ ! -s "${path}" ]]; then
        echo "${message}:" >&2
        echo "${path}" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    local message="${2:-Required directory is missing}"
    if [[ ! -d "${path}" ]]; then
        echo "${message}:" >&2
        echo "${path}" >&2
        exit 1
    fi
}

pretty_json() {
    if command -v jq >/dev/null 2>&1; then jq .; else cat; fi
}

path_bytes() {
    local path="$1"
    if [[ -e "${path}" ]]; then du -sb "${path}" | awk '{print $1}'; else echo 0; fi
}

profile_dictionary_args=()
collect_profile_dictionaries() {
    local dictionary
    profile_dictionary_args=()
    for dictionary in \
        "${profile_dir}/base_dictionary.txt" \
        "${profile_dir}/custom_words.txt"; do
        if [[ -s "${dictionary}" ]]; then
            profile_dictionary_args+=(--dictionary "${dictionary}")
        fi
    done
    if [[ ${#profile_dictionary_args[@]} -eq 0 ]]; then
        echo "No pronunciation dictionary was found in ${profile_dir}. Run Stage 1 first." >&2
        exit 1
    fi
}

stage_header() {
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

# ============================================================
# Stage 0: Build patched Docker image
# ============================================================
if (( stage <= 0 && stop_stage >= 0 )); then
    stage_header "Stage 0: Build patched Docker image"

    mkdir -p "${root_dir}/docker_patch"
    profile_definition="${root_dir}/etc/profiles/${profile}.yml"
    patched_definition="${root_dir}/docker_patch/${profile}.yml"

    if [[ -s "${profile_definition}" ]]; then
        cp "${profile_definition}" "${patched_definition}"
    else
        echo "Profile definition is missing from this checkout; trying tag v2.1."
        git -C "${root_dir}" fetch --tags
        git -C "${root_dir}" show "v2.1:etc/profiles/${profile}.yml" \
            > "${patched_definition}"
    fi
    require_file "${patched_definition}" "Failed to obtain the profile definition"

    cat > "${root_dir}/docker_patch/Dockerfile" <<DOCKER_EOF
FROM synesthesiam/voice2json:latest
COPY ${profile}.yml /usr/lib/voice2json/etc/profiles/${profile}.yml
DOCKER_EOF

    docker build -t "${image}" "${root_dir}/docker_patch"
    docker run --rm --entrypoint bash "${image}" \
        -lc "test -s /usr/lib/voice2json/etc/profiles/${profile}.yml"
fi

# ============================================================
# Stage 1: Download Mandarin profile and prepare MAC-SLU JSONL
# ============================================================
if (( stage <= 1 && stop_stage >= 1 )); then
    stage_header "Stage 1: Download profile and prepare MAC-SLU JSONL"

    require_file "${prepare_py}" "MAC-SLU preparation script was not found"
    v2j --debug download-profile
    require_dir "${profile_dir}" "Profile directory was not created"

    python "${prepare_py}" \
        --repo-id "${repo_id}" \
        --download-dir "${root_dir}/${download_dir}" \
        --extract-root "${root_dir}/${extract_root}" \
        --jsonl-root "${root_dir}/${json_root}" \
        --splits train dev test

    require_file "${root_dir}/${json_root}/train.jsonl" "Training manifest was not created"
fi

# ============================================================
# Stage 2: Generate and load external grammar
# ============================================================
if (( stage <= 2 && stop_stage >= 2 )); then
    stage_header "Stage 2: Generate and load MAC-SLU grammar"

    train_jsonl="${root_dir}/${json_root}/train.jsonl"
    require_file "${train_jsonl}" "MAC-SLU training manifest is missing; run Stage 1 first"
    require_dir "${profile_dir}" "Profile is missing; run Stage 1 first"
    collect_profile_dictionaries

    grammar_cmd=(
        python "${root_dir}/local/build_macslu_grammar.py"
        --train-jsonl "${train_jsonl}"
        --grammar-out "${grammar_path}"
        --intent-map-out "${intent_map_path}"
        --stats-out "${grammar_stats_path}"
        --max-entries "${max_grammar_entries}"
    )
    grammar_cmd+=("${profile_dictionary_args[@]}")
    if [[ "${disable_pos_lexicon}" == true ]]; then
        grammar_cmd+=(--disable-pos-lexicon)
    fi
    "${grammar_cmd[@]}"

    require_file "${grammar_path}" "Generated grammar is empty"
    require_file "${intent_map_path}" "Generated intent map is empty"

    if [[ -f "${profile_dir}/sentences.ini" && ! -f "${profile_dir}/sentences.ini.original" ]]; then
        cp "${profile_dir}/sentences.ini" "${profile_dir}/sentences.ini.original"
    fi
    cp "${grammar_path}" "${profile_dir}/sentences.ini"

    rm -f \
        "${profile_dir}/language_model.txt" \
        "${profile_dir}/language_model.fst" \
        "${profile_dir}/intent.fst" \
        "${profile_dir}/intent.pickle.gz" \
        "${profile_dir}/dictionary.txt" \
        "${profile_dir}/unknown_words.txt"

    echo "Grammar source: ${grammar_path}"
    echo "Grammar destination: ${profile_dir}/sentences.ini"
    pretty_json < "${grammar_stats_path}"
fi

# ============================================================
# Stage 3: Compile grammar and train profile
# ============================================================
if (( stage <= 3 && stop_stage >= 3 )); then
    stage_header "Stage 3: Train profile"
    require_file "${profile_dir}/sentences.ini" "Profile grammar is missing; run Stage 2 first"
    v2j --debug train-profile
    ls -lh \
        "${profile_dir}/language_model.txt" \
        "${profile_dir}/dictionary.txt" \
        "${profile_dir}/intent.pickle.gz"
fi

# ============================================================
# Stage 4: Text-only intent smoke test
# ============================================================
if (( stage <= 4 && stop_stage >= 4 )); then
    stage_header "Stage 4: Text intent smoke test"

    if [[ -z "${test_text}" ]]; then
        test_text="$(python - "${root_dir}/${json_root}/train.jsonl" <<'PY'
import json
import sys
from pathlib import Path
for line in Path(sys.argv[1]).open(encoding="utf-8"):
    if line.strip():
        print(json.loads(line).get("query", ""))
        break
PY
)"
    fi
    [[ -n "${test_text}" ]] || { echo "No smoke-test text is available" >&2; exit 1; }
    echo "Input text: ${test_text}"
    printf '%s\n' "${test_text}" \
        | v2j recognize-intent --text-input \
        | tee "${exp_dir}/text_intent.json" \
        | pretty_json
fi

# ============================================================
# Stage 5: Dataset and grammar diagnostics
# ============================================================
if (( stage <= 5 && stop_stage >= 5 )); then
    stage_header "Stage 5: MAC-SLU grammar diagnostics"
    require_file "${grammar_path}" "Grammar is missing; run Stage 2 first"
    require_file "${intent_map_path}" "Intent map is missing; run Stage 2 first"
    require_file "${grammar_stats_path}" "Grammar statistics are missing; run Stage 2 first"

    python - "${grammar_path}" "${intent_map_path}" "${grammar_stats_path}" <<'PY'
import configparser
import json
import sys
from pathlib import Path

grammar_path, map_path, stats_path = map(Path, sys.argv[1:])
parser = configparser.ConfigParser(
    allow_no_value=True,
    delimiters=("=",),
    interpolation=None,
    strict=False,
)
parser.optionxform = str
parser.read(grammar_path, encoding="utf-8")
intent_map = json.loads(map_path.read_text(encoding="utf-8"))
stats = json.loads(stats_path.read_text(encoding="utf-8"))
missing = sorted(set(intent_map) - set(parser.sections()))
if missing:
    raise SystemExit(f"Intent-map sections missing from grammar: {missing[:5]}")
print(json.dumps({
    "grammar_sections": len(parser.sections()),
    "intent_sections": len(intent_map),
    "slot_sections": stats.get("slot_sections", 0),
    "grammar_entries": stats.get("grammar_entries", 0),
    "noun_values": stats.get("noun_values", 0),
    "verb_values": stats.get("verb_values", 0),
    "zero_intent_rows": stats.get("zero_intent_rows", 0),
    "multi_intent_rows": stats.get("multi_intent_rows", 0),
}, ensure_ascii=False, indent=2))
PY
fi

# ============================================================
# Stage 6: MAC-SLU inference
# ============================================================
if (( stage <= 6 && stop_stage >= 6 )); then
    stage_header "Stage 6: MAC-SLU inference"
    require_file "${intent_map_path}" "Intent map is missing; run Stage 2 first"
    collect_profile_dictionaries

    for test_set in ${test_sets}; do
        test_jsonl="${root_dir}/${json_root}/${test_set}.jsonl"
        output_dir="${exp_dir}/${test_set}"
        pred_file="${output_dir}/predictions.jsonl"
        require_file "${test_jsonl}" "MAC-SLU ${test_set} manifest is missing"
        mkdir -p "${output_dir}"

        infer_cmd=(
            python "${root_dir}/local/infer_macslu_voice2json.py"
            --input-jsonl "${test_jsonl}"
            --output-jsonl "${pred_file}"
            --intent-map "${intent_map_path}"
            --image "${image}"
            --mount "${root_dir}"
            --mount "${qwen3_slu_root}"
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
# Stage 7: Evaluation and resource report
# ============================================================
if (( stage <= 7 && stop_stage >= 7 )); then
    stage_header "Stage 7: Evaluation and resource report"
    require_file "${metrics_py}" "MAC-SLU metrics script was not found"

    for test_set in ${test_sets}; do
        pred_file="${exp_dir}/${test_set}/predictions.jsonl"
        gt_file="${root_dir}/${json_root}/${test_set}.jsonl"
        output_dir="${exp_dir}/${test_set}"
        require_file "${gt_file}" "MAC-SLU ${test_set} manifest is missing"
        if [[ ! -s "${pred_file}" ]]; then
            echo "Prediction file is missing; skipping ${test_set}: ${pred_file}" >&2
            continue
        fi
        python "${metrics_py}" \
            --output_dir "${output_dir}" \
            "${pred_file}" "${gt_file}" \
            | tee "${output_dir}/metrics.txt"
    done

    if [[ "${skip_resource_report}" == false ]]; then
        require_dir "${profile_dir}" "Profile directory is missing"
        test_jsonl="${root_dir}/${json_root}/test.jsonl"
        require_file "${test_jsonl}" "MAC-SLU test manifest is missing"
        resource_dir="${exp_dir}/resource"
        report_file="${resource_dir}/resource_report.txt"
        mkdir -p "${resource_dir}"

        first_audio="$(python "${root_dir}/local/model_resource_utils.py" first-audio --jsonl "${test_jsonl}")"
        require_file "${first_audio}" "Could not resolve a test audio file"

        ram_container="voice2json_ram_${$}_${RANDOM}"
        ram_stats_file="$(mktemp)"
        ram_input_file="$(mktemp)"
        cleanup() {
            docker rm -f "${ram_container}" >/dev/null 2>&1 || true
            rm -f "${ram_stats_file}" "${ram_input_file}"
        }
        trap cleanup EXIT
        for _ in $(seq 1 "${ram_repeat}"); do echo "${first_audio}"; done > "${ram_input_file}"

        docker run --rm -i \
            --name "${ram_container}" \
            --init \
            -v "${HOME}:${HOME}" \
            -v "${root_dir}:${root_dir}" \
            -v "/dev/shm:/dev/shm" \
            -w "${root_dir}" \
            -e "HOME=${HOME}" \
            --user "$(id -u):$(id -g)" \
            "${image}" --profile "${profile}" \
            transcribe-wav --stdin-files \
            < "${ram_input_file}" >/dev/null 2>&1 &
        ram_pid=$!

        for _ in $(seq 1 100); do
            docker inspect "${ram_container}" >/dev/null 2>&1 && break
            sleep 0.05
        done
        while kill -0 "${ram_pid}" >/dev/null 2>&1; do
            docker stats --no-stream --format '{{.MemUsage}}' "${ram_container}" 2>/dev/null \
                | awk -F' / ' '{print $1}' >> "${ram_stats_file}" || true
            sleep 0.1
        done
        wait "${ram_pid}"

        peak_ram_bytes="$(python "${root_dir}/local/model_resource_utils.py" peak-ram --stats-file "${ram_stats_file}")"
        profile_bytes="$(path_bytes "${profile_dir}")"
        image_bytes="$(docker image inspect "${image}" --format '{{.Size}}')"
        grammar_bytes="$(path_bytes "${grammar_path}")"

        {
            echo "voice2json MAC-SLU Resource Report"
            echo "Profile: ${profile}"
            echo "Image: ${image}"
            echo
            echo "Disk size (bytes)"
            echo "  Profile: ${profile_bytes}"
            echo "  Docker image: ${image_bytes}"
            echo "  Generated grammar: ${grammar_bytes}"
            echo
            echo "Peak container RAM (bytes): ${peak_ram_bytes}"
            echo
            echo "Model family: Pocketsphinx GMM-HMM"
            echo "Neural parameter count: N/A"
        } | tee "${report_file}"
        cleanup
        trap - EXIT
    fi
fi
