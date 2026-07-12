#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR='/home/yuzichen'
PROJECT_DIR="${ROOT_DIR}/assignment_A/sft"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/prepare_code_sft_data.py" \
  --data_dir "${ROOT_DIR}/python_code_instructions_18k_alpaca" \
  --output_dir "${PROJECT_DIR}/data" \
  --min_output_len 20 \
  --max_output_len 4096 \
  --remove_duplicates True \
  --check_syntax True

