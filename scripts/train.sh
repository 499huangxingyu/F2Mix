#!/bin/bash
# F2Mix training script.
#
# Usage:
#   bash scripts/train.sh <BACKBONE> <DATA_PATH> [GPU]
# Example:
#   bash scripts/train.sh densenet your/path/to/dataset.hdf5 0

set -e

BACKBONE=${1:-densenet}
DATA_PATH=${2:-"your/path/to/dataset.hdf5"}
GPU=${3:-0}
SAVE_DIR=${4:-"./results"}

nohup python -u main.py \
    --mode train \
    --backbone "${BACKBONE}" \
    --data_path "${DATA_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --gpu "cuda:${GPU}" \
    > "${SAVE_DIR}_${BACKBONE}_train.out" 2>&1 &

echo "========================================================"
echo "Training ${BACKBONE} started; output is saved to ${SAVE_DIR}_${BACKBONE}_train.out"
echo "========================================================"
tail -f "${SAVE_DIR}_${BACKBONE}_train.out"
