#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# voice2json Docker Environment Setup
#
# This script:
#   1. Checks the Linux/Docker environment
#   2. Creates the project directory
#   3. Creates a voice2json Docker wrapper
#   4. Downloads the Docker image
#   5. Verifies that voice2json starts correctly
# ============================================================

PROJECT_DIR="${HOME}/projects/voice2json-poc"
BIN_DIR="${PROJECT_DIR}/bin"
IMAGE_NAME="synesthesiam/voice2json"
WRAPPER_PATH="${BIN_DIR}/voice2json"

log() {
    printf '\n===== %s =====\n' "$1"
}

fail() {
    printf '\nERROR: %s\n' "$1" >&2
    exit 1
}

# ------------------------------------------------------------
# 1. Check operating system
# ------------------------------------------------------------

log "Checking Operating System"

if [[ "$(uname -s)" != "Linux" ]]; then
    fail "This setup script currently supports Linux only."
fi

echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"

if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    echo "Distribution: ${PRETTY_NAME:-unknown}"
fi

# ------------------------------------------------------------
# 2. Check CPU architecture
# ------------------------------------------------------------

log "Checking CPU Architecture"

ARCH="$(uname -m)"

case "${ARCH}" in
    x86_64|amd64)
        echo "Supported architecture: ${ARCH}"
        ;;
    *)
        fail "Unsupported CPU architecture: ${ARCH}. This script expects x86_64/amd64."
        ;;
esac

# ------------------------------------------------------------
# 3. Check Docker
# ------------------------------------------------------------

log "Checking Docker"

if ! command -v docker >/dev/null 2>&1; then
    fail "Docker is not installed."
fi

docker --version

if ! docker info >/dev/null 2>&1; then
    fail "Docker is installed, but the Docker engine is unavailable."
fi

echo "Docker engine is running."

# ------------------------------------------------------------
# 4. Create project directories
# ------------------------------------------------------------

log "Creating Project Directory"

mkdir -p "${BIN_DIR}"
mkdir -p "${PROJECT_DIR}/profiles"
mkdir -p "${PROJECT_DIR}/audio"
mkdir -p "${PROJECT_DIR}/scripts"
mkdir -p "${PROJECT_DIR}/output"

echo "Project directory: ${PROJECT_DIR}"

# ------------------------------------------------------------
# 5. Create Docker wrapper
# ------------------------------------------------------------

log "Creating voice2json Wrapper"

cat > "${WRAPPER_PATH}" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="synesthesiam/voice2json"

docker run --rm -i \
    --init \
    -v "${HOME}:${HOME}" \
    -v "/dev/shm:/dev/shm" \
    -w "$(pwd)" \
    -e "HOME=${HOME}" \
    --user "$(id -u):$(id -g)" \
    "${IMAGE_NAME}" "$@"
WRAPPER

chmod +x "${WRAPPER_PATH}"

echo "Wrapper created at:"
echo "${WRAPPER_PATH}"

# ------------------------------------------------------------
# 6. Pull Docker image
# ------------------------------------------------------------

log "Downloading Docker Image"

docker pull "${IMAGE_NAME}"

# ------------------------------------------------------------
# 7. Inspect Docker image
# ------------------------------------------------------------

log "Inspecting Docker Image"

docker image inspect "${IMAGE_NAME}" \
    --format 'Image ID: {{.Id}}
Architecture: {{.Architecture}}
Operating System: {{.Os}}
Size: {{.Size}} bytes'

IMAGE_ARCH="$(
    docker image inspect "${IMAGE_NAME}" \
        --format '{{.Architecture}}'
)"

if [[ "${IMAGE_ARCH}" != "amd64" ]]; then
    fail "Unexpected Docker image architecture: ${IMAGE_ARCH}"
fi

# ------------------------------------------------------------
# 8. Test voice2json
# ------------------------------------------------------------

log "Testing voice2json"

cd "${PROJECT_DIR}"

HELP_OUTPUT="$("${WRAPPER_PATH}" --help 2>&1)" || {
    echo "${HELP_OUTPUT:-}"
    fail "voice2json failed to start."
}

echo "${HELP_OUTPUT}" | head -n 50

if ! grep -q "train-profile" <<< "${HELP_OUTPUT}"; then
    fail "voice2json started, but the expected train-profile command was not found."
fi

if ! grep -q "transcribe-wav" <<< "${HELP_OUTPUT}"; then
    fail "voice2json started, but the expected transcribe-wav command was not found."
fi

if ! grep -q "recognize-intent" <<< "${HELP_OUTPUT}"; then
    fail "voice2json started, but the expected recognize-intent command was not found."
fi

# ------------------------------------------------------------
# 9. Create convenient shell launcher
# ------------------------------------------------------------

log "Creating Environment File"

cat > "${PROJECT_DIR}/env.sh" <<EOF
#!/usr/bin/env bash

export VOICE2JSON_PROJECT="${PROJECT_DIR}"
export PATH="${BIN_DIR}:\${PATH}"

cd "${PROJECT_DIR}"
EOF

chmod +x "${PROJECT_DIR}/env.sh"

# ------------------------------------------------------------
# 10. Final diagnostics
# ------------------------------------------------------------

log "Final Diagnostics"

echo "Project directory:"
echo "  ${PROJECT_DIR}"

echo
echo "Wrapper:"
ls -l "${WRAPPER_PATH}"

echo
echo "Docker image:"
docker image inspect "${IMAGE_NAME}" \
    --format '  Architecture={{.Architecture}}
  OS={{.Os}}
  Size={{.Size}} bytes'

echo
echo "Project structure:"
find "${PROJECT_DIR}" \
    -maxdepth 2 \
    -type d \
    -printf '  %p\n' \
    | sort

# ------------------------------------------------------------
# Completed
# ------------------------------------------------------------

log "Setup Completed Successfully"

cat <<EOF
Run the following commands to enter the project:

    source "${PROJECT_DIR}/env.sh"

Then verify voice2json with:

    voice2json --help

Your project directory is:

    ${PROJECT_DIR}
EOF
SCRIPT