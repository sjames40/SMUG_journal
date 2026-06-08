#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
ROOT="/egr/research-slim/sunyongl/SMUG/SMUG_journal-main"
PYTHON_BIN="/egr/research-slim/sunyongl/miniconda3/envs/pytorch-happysun/bin/python"
DATA_ROOT="$ROOT/data/NEW_KSPACE"
MASK_ROOT="$ROOT/data/MASK_4X"
GPU_IDS="${GPU_IDS:-0}"
TRAIN_SIZE="${TRAIN_SIZE:-3000}"
VALI_SIZE="${VALI_SIZE:-32}"
TEST_SIZE="${TEST_SIZE:-64}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
BLOCK_ITER="${BLOCK_ITER:-8}"
SMOOTHING_EPSILON="${SMOOTHING_EPSILON:-0.01}"
NUM_SAMPLE="${NUM_SAMPLE:-10}"
LOSS_LAMBDA="${LOSS_LAMBDA:-1.0}"
N_RES_BLOCKS="${N_RES_BLOCKS:-3}"
CHECKPOINT_DIR="$ROOT/checkpoints"
PRETRAIN_NAME="${PRETRAIN_NAME:-pretrain_denoiser}"
VANILLA_NAME="${VANILLA_NAME:-vanilla_modl}"
SMUG_NAME="${SMUG_NAME:-smug_modl}"
PRETRAIN_CKPT="$CHECKPOINT_DIR/$PRETRAIN_NAME/vali_best.pth"
SMUG_CKPT="$CHECKPOINT_DIR/$SMUG_NAME/vali_best.pth"

case "$MODE" in
  generate_mask)
    "$PYTHON_BIN" "$ROOT/generate_base_mask.py" --dataroot "$DATA_ROOT" --output-dir "$MASK_ROOT"
    ;;
  pretrain)
    cd "$ROOT"
    "$PYTHON_BIN" pretrain_denoiser.py --dataroot "$DATA_ROOT" --mask_dataroot "$MASK_ROOT" --gpu_ids "$GPU_IDS" --trainSize "$TRAIN_SIZE" --valiSize "$VALI_SIZE" --batchSize "$BATCH_SIZE" --epoch "$EPOCHS" --lr "$LR" --smoothing_epsilon "$SMOOTHING_EPSILON" --n_res_blocks "$N_RES_BLOCKS" --checkpoints_dir "$CHECKPOINT_DIR" --name "$PRETRAIN_NAME"
    ;;
  vanilla)
    cd "$ROOT"
    "$PYTHON_BIN" train_vanilla_MoDL.py --dataroot "$DATA_ROOT" --mask_dataroot "$MASK_ROOT" --gpu_ids "$GPU_IDS" --trainSize "$TRAIN_SIZE" --valiSize "$VALI_SIZE" --batchSize "$BATCH_SIZE" --epoch "$EPOCHS" --lr "$LR" --blockIter "$BLOCK_ITER" --n_res_blocks "$N_RES_BLOCKS" --checkpoints_dir "$CHECKPOINT_DIR" --name "$VANILLA_NAME"
    ;;
  smug)
    cd "$ROOT"
    "$PYTHON_BIN" train_SMUG.py --dataroot "$DATA_ROOT" --mask_dataroot "$MASK_ROOT" --netGpath "$PRETRAIN_CKPT" --gpu_ids "$GPU_IDS" --trainSize "$TRAIN_SIZE" --valiSize "$VALI_SIZE" --batchSize "$BATCH_SIZE" --epoch "$EPOCHS" --lr "$LR" --blockIter "$BLOCK_ITER" --num_sample "$NUM_SAMPLE" --smoothing_epsilon "$SMOOTHING_EPSILON" --n_res_blocks "$N_RES_BLOCKS" --LossLambda "$LOSS_LAMBDA" --checkpoints_dir "$CHECKPOINT_DIR" --name "$SMUG_NAME"
    ;;
  test)
    cd "$ROOT"
    "$PYTHON_BIN" test.py --dataroot "$DATA_ROOT" --mask_dataroot "$MASK_ROOT" --netGpath "$SMUG_CKPT" --gpu_ids "$GPU_IDS" --train_valiSize "$((TRAIN_SIZE + VALI_SIZE))" --testSize "$TEST_SIZE" --blockIter "$BLOCK_ITER" --smoothing SMUG --num_sample "$NUM_SAMPLE" --smoothing_epsilon "$SMOOTHING_EPSILON" --n_res_blocks "$N_RES_BLOCKS"
    ;;
  all)
    "$0" generate_mask
    "$0" pretrain
    "$0" vanilla
    "$0" smug
    "$0" test
    ;;
  *)
    echo "usage: $0 [generate_mask|pretrain|vanilla|smug|test|all]" >&2
    exit 1
    ;;
esac
