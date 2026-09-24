#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  -m ddp_gradient_compression.train \
  --hook topk \
  --topk-ratio 0.01 \
  --download
