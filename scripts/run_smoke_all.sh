#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAIN_SUBSET="${TRAIN_SUBSET:-1024}"
TEST_SUBSET="${TEST_SUBSET:-512}"

cd "${PROJECT_ROOT}"

for hook in dense fp16 topk powersgd; do
  hook_args=()

  case "${hook}" in
    topk)
      hook_args+=(--topk-ratio 0.01)
      ;;
    powersgd)
      hook_args+=(--powersgd-rank 1 --powersgd-start-iter 10)
      ;;
  esac

  echo "Running ${hook} smoke test..."
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    -m ddp_gradient_compression.train \
    --dataset synthetic \
    --hook "${hook}" \
    --train-subset "${TRAIN_SUBSET}" \
    --test-subset "${TEST_SUBSET}" \
    --epochs 1 \
    "${hook_args[@]}"
done

echo "All smoke tests passed."
