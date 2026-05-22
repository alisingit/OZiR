#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${LAB_DIR}/.." && pwd)"
IMAGE_NAME="lab4-system-tts"

docker build -f "${LAB_DIR}/docker/system-tts.Dockerfile" -t "${IMAGE_NAME}" "${LAB_DIR}"

docker run --rm \
  -v "${LAB_DIR}:/work" \
  "${IMAGE_NAME}" \
  python3 /work/src/generate_audio_runs.py --engine rhvoice --voice "" --overwrite

docker run --rm \
  -v "${LAB_DIR}:/work" \
  "${IMAGE_NAME}" \
  python3 /work/src/generate_audio_runs.py --engine espeak_ng --voice ru --overwrite

printf 'System TTS audio saved under %s/audio/{rhvoice,espeak_ng}\n' "${LAB_DIR}"
