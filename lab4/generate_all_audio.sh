#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${LAB_DIR}/.." && pwd)"
PIPER_MODEL="${LAB_DIR}/models/piper/ru_RU-irina-medium.onnx"
PIPER_CONFIG="${LAB_DIR}/models/piper/ru_RU-irina-medium.onnx.json"

mkdir -p "${LAB_DIR}/models/piper" "${LAB_DIR}/.cache"

if [[ ! -f "${PIPER_MODEL}" ]]; then
  curl -L --fail \
    -o "${PIPER_MODEL}" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx"
fi

if [[ ! -f "${PIPER_CONFIG}" ]]; then
  curl -L --fail \
    -o "${PIPER_CONFIG}" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json"
fi

export XDG_CACHE_HOME="${LAB_DIR}/.cache/xdg"
export TORCH_HOME="${LAB_DIR}/.cache/torch"
export HF_HOME="${LAB_DIR}/.cache/huggingface"
export TTS_HOME="${LAB_DIR}/.cache/tts"
export COQUI_TOS_AGREED=1

"${LAB_DIR}/.venv/bin/python" "${LAB_DIR}/src/generate_audio_runs.py" \
  --engine silero \
  --voice baya \
  --overwrite

"${LAB_DIR}/.venv/bin/python" "${LAB_DIR}/src/generate_audio_runs.py" \
  --engine piper \
  --voice irina \
  --piper-model "${PIPER_MODEL}" \
  --overwrite

"${LAB_DIR}/src/generate_system_tts_docker.sh"

"${LAB_DIR}/.venv-coqui/bin/python" "${LAB_DIR}/src/generate_audio_runs.py" \
  --engine coqui_xtts \
  --voice xtts_v2 \
  --coqui-model-dir "${LAB_DIR}/models/coqui_xtts/XTTS-v2" \
  --coqui-speaker-wav "${LAB_DIR}/audio/rhvoice/stress/01.wav" \
  --overwrite

python3 "${LAB_DIR}/src/combine_audio_manifests.py"

printf 'All audio files are saved under %s/audio\n' "${LAB_DIR}"
