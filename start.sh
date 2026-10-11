#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    --max_restarts=0 \
    start_train.py \
    --config cfg/demo/mlm_train.yaml \
    "$@"