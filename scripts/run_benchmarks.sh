#!/usr/bin/env bash

# Benchmark 任一步失败都停止，避免把不完整结果继续写入报告。
set -euo pipefail

# 使用绝对仓库路径，保证 CSV 总是写入当前项目的 results/raw。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
csv_path="${repo_root}/results/raw/transpose_v2.csv"

# 选择方阵、长条、宽条和非 32 整除大矩阵，观察 Shape 对跨步写和 Shared Memory Tiling 的影响。
shapes=(
    "1024x8192"
    "8192x1024"
    "4096x4096"
    "4097x3073"
)

# 每个 Shape 在同一进程依次测 V0 Copy、V1 Naive 和 V2 Tiled，保持统计方法完全一致。
for shape in "${shapes[@]}"; do
    "${repo_root}/build/transpose_bench" \
        --kernel all \
        --shape "${shape}" \
        --warmup 20 \
        --iterations 100 \
        --groups 5 \
        --csv "${csv_path}"
done
