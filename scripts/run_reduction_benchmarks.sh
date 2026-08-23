#!/usr/bin/env bash

# 任意 Benchmark 失败时立即停止，避免保留不完整的正式数据。
set -euo pipefail

# 从脚本位置解析仓库根目录，使输出路径不依赖当前工作目录。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
csv_path="${repo_root}/results/raw/reduction_v0.csv"

# 覆盖 Warp、Block、非 2 的幂、非整除和大规模多阶段归约。
sizes=(
    1
    31
    32
    33
    255
    256
    257
    1023
    1024
    1025
    1000003
    16777219
)

# 每个 N 使用相同的 deterministic random 输入和统一统计方法。
for input_count in "${sizes[@]}"; do
    "${repo_root}/build/reduction_bench" \
        --kernel interleaved \
        --size "${input_count}" \
        --warmup 20 \
        --iterations 100 \
        --groups 5 \
        --csv "${csv_path}"
done
