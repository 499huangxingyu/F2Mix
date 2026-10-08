#!/bin/bash
# F2Mix testing script.
#
# Usage:
#   bash scripts/test.sh <BACKBONE> <DATA_PATH> <TEST_MODEL> [GPU]
# Example:
#   bash scripts/test.sh densenet your/path/to/dataset.hdf5 \
#        results/densenet/models/best_model.pth 0

set -e

BACKBONE=${1:-densenet}
DATA_PATH=${2:-"your/path/to/dataset.hdf5"}
TEST_MODEL=${3:-"results/${BACKBONE}/models/best_model.pth"}
GPU=${4:-0}
SAVE_DIR=${5:-"./results"}

python -u main.py \
    --mode test \
    --backbone "${BACKBONE}" \
    --data_path "${DATA_PATH}" \
    --test_model "${TEST_MODEL}" \
    --save_dir "${SAVE_DIR}" \
    --gpu "cuda:${GPU}"
