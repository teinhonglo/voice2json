#!/usr/bin/env bash
set -euo pipefail

docker run --rm -i \
    --init \
    -v "${HOME}:${HOME}" \
    -v "$(pwd)/wavs:${HOME}/wavs:ro" \
    -v "/dev/shm:/dev/shm" \
    -w "${HOME}" \
    -e "HOME=${HOME}" \
    --user "$(id -u):$(id -g)" \
    "${VOICE2JSON_IMAGE:-voice2json-zh:local}" "$@"
