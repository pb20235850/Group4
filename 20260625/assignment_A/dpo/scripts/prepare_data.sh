#!/bin/bash

export SEED=${SEED:-42}
export TEST_SIZE=${TEST_SIZE:-500}
export MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
export SOURCE_DIR=${SOURCE_DIR:-source_data}
export OUTPUT_DIR=${OUTPUT_DIR:-"dpo/data"}

python dpo/prepare_dpo_data.py
