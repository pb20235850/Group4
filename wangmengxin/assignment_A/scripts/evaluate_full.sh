#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SFT_DIR}/.." && pwd)"
DEFAULT_PYTHON="/home/wangmengxin/assignment_A/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" sft/scripts/evaluate_code_predictions.py \
  --predictions "${PREDICTIONS:-sft/outputs/qwen15_code_full_predict/generated_predictions.jsonl}" \
  --mbpp_dir "${MBPP_DIR:-/home/wangmengxin/mbpp}" \
  --config "${MBPP_CONFIG:-sanitized}" \
  --split "${MBPP_SPLIT:-test}" \
  --output_dir "${EVAL_DIR:-sft/outputs/eval_mbpp}" \
  --limit "${LIMIT:-0}" \
  --test_timeout "${TEST_TIMEOUT:-5.0}" \
  --memory_mb "${MEMORY_MB:-1024}"