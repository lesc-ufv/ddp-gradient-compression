#!/usr/bin/env bash
set -euo pipefail

# Run the same script on all participating nodes.
#
# Adjust:
#   RDZV_ENDPOINT
#   GLOO_SOCKET_IFNAME
#   DATA_DIR
#
# Example:
#   node0 = 10.0.0.10
#   node1 = 10.0.0.11
#   node2 = 10.0.0.12
#   node3 = 10.0.0.13

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-10}"

# Uncomment and change if the host has multiple network interfaces.
# export GLOO_SOCKET_IFNAME=enp1s0

RDZV_ENDPOINT="${RDZV_ENDPOINT:-10.0.0.10:29500}"
DATA_DIR="${DATA_DIR:-/data/mnist}"

torchrun \
  --nnodes=4 \
  --nproc-per-node=1 \
  --rdzv-id=ddp-gradient-compression-001 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${RDZV_ENDPOINT}" \
  -m ddp_gradient_compression.train \
  --hook topk \
  --topk-ratio 0.01 \
  --data-dir "${DATA_DIR}"
