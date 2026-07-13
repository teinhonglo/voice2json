#!/usr/bin/env bash
set -euo pipefail

docker_args=(
    --rm
    -i
    --init
    -v "${HOME}:${HOME}"
    -v "$(pwd):$(pwd)"
    -v "$(pwd)/wavs:${HOME}/wavs:ro"
    -v "/dev/shm:/dev/shm"
    -w "$(pwd)"
    -e "HOME=${HOME}"
    --user "$(id -u):$(id -g)"
)

for mount_path in ${VOICE2JSON_MOUNTS:-}; do
    if [ -d "${mount_path}" ]; then
        docker_args+=(-v "${mount_path}:${mount_path}")
    fi
done

docker run "${docker_args[@]}" \
    "${VOICE2JSON_IMAGE:-voice2json-zh:local}" "$@"
