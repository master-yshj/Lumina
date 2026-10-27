#!/usr/bin/env bash
# AI 编写：默认运行随仓库提供的 MLM demo；额外 CLI 覆盖原样传递。
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
