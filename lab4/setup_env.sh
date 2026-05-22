#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3.11 -m venv "${LAB_DIR}/.venv"
"${LAB_DIR}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"${LAB_DIR}/.venv/bin/python" -m pip install -r "${LAB_DIR}/requirements-python.txt"

python3.10 -m venv "${LAB_DIR}/.venv-coqui"
"${LAB_DIR}/.venv-coqui/bin/python" -m pip install --upgrade "pip" "setuptools<81" wheel
"${LAB_DIR}/.venv-coqui/bin/python" -m pip install \
  "cmake" \
  "numpy<1.27" \
  "llvmlite==0.41.1" \
  "numba==0.58.1" \
  "torch==2.2.2" \
  "torchaudio==2.2.2"
"${LAB_DIR}/.venv-coqui/bin/python" -m pip install "TTS==0.22.0"
"${LAB_DIR}/.venv-coqui/bin/python" -m pip install "transformers==4.40.2" "tokenizers<0.20"

mkdir -p "${LAB_DIR}/.cache" "${LAB_DIR}/models/piper"

printf 'Lab4 environments are ready:\n'
printf '  %s/.venv\n' "${LAB_DIR}"
printf '  %s/.venv-coqui\n' "${LAB_DIR}"
