#!/usr/bin/env bash
set -euo pipefail

# Run once per host. Defaults: four hosts, one rank and one dedicated U55C per host.
# NODE_RANK must be 0, 1, 2 or 3, matching the global order in ACCL_CONFIG.
: "${NODE_RANK:?Set NODE_RANK to this host's index in ACCL_CONFIG}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 host's CPU-network address}"
: "${ACCL_CONFIG:?Set ACCL_CONFIG to the cluster JSON configuration}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# Resolve before changing directories so relative config paths remain valid.
export ACCL_CONFIG="$(realpath -- "${ACCL_CONFIG}")"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
cd "${PROJECT_ROOT}"

torchrun \
  --nnodes="${NNODES:-4}" \
  --nproc-per-node="${NPROC_PER_NODE:-1}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT:-29500}" \
  -m ddp_gradient_compression.train \
  --backend accl \
  --dataset synthetic \
  --hook "${HOOK:-dense}" \
  --train-subset 4096 \
  --test-subset 512 \
  --epochs 1 \
  "$@"
