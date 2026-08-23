#!/usr/bin/env bash

# Benchmark 任一步失败都停止，避免把不完整结果继续写入报告。
set -euo pipefail

# 使用绝对仓库路径，保证 CSV 总是写入当前项目的 results/raw。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 按提示词的正式默认值运行 20 次预热、每组 100 次 Kernel、共 5 组。
"${repo_root}/build/transpose_bench" \
    --kernel copy \
    --shape 4096x4096 \
    --warmup 20 \
    --iterations 100 \
    --groups 5 \
    --csv "${repo_root}/results/raw/transpose_v0.csv"
