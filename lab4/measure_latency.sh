#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="lab4-system-tts"
LATENCY_PATH="${LAB_DIR}/data/latency.csv"
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

rm -f "${LATENCY_PATH}"

"${LAB_DIR}/.venv/bin/python" "${LAB_DIR}/src/measure_latency.py" \
  --engine silero \
  --voice baya \
  --output "${LATENCY_PATH}"

docker build -f "${LAB_DIR}/docker/system-tts.Dockerfile" -t "${IMAGE_NAME}" "${LAB_DIR}"

docker run --rm \
  -v "${LAB_DIR}:/work" \
  "${IMAGE_NAME}" \
  python3 /work/src/measure_latency.py \
    --engine rhvoice \
    --voice "" \
    --output /work/data/latency.csv \
    --append

"${LAB_DIR}/.venv/bin/python" "${LAB_DIR}/src/measure_latency.py" \
  --engine piper \
  --voice irina \
  --piper-model "${PIPER_MODEL}" \
  --output "${LATENCY_PATH}" \
  --append

"${LAB_DIR}/.venv-coqui/bin/python" "${LAB_DIR}/src/measure_latency.py" \
  --engine coqui_xtts \
  --voice xtts_v2 \
  --coqui-model-dir "${LAB_DIR}/models/coqui_xtts/XTTS-v2" \
  --coqui-speaker-wav "${LAB_DIR}/audio/rhvoice/stress/01.wav" \
  --output "${LATENCY_PATH}" \
  --append

docker run --rm \
  -v "${LAB_DIR}:/work" \
  "${IMAGE_NAME}" \
  python3 /work/src/measure_latency.py \
    --engine espeak_ng \
    --voice ru \
    --output /work/data/latency.csv \
    --append

python3 "${LAB_DIR}/src/analyze_results.py"
printf 'Latency measurements saved to %s\n' "${LATENCY_PATH}"
