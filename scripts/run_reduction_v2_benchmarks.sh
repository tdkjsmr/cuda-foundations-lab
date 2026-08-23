#!/usr/bin/env bash

# 任意 Benchmark 失败时立即停止，避免保留不完整的正式数据。
set -euo pipefail

# 从脚本位置解析仓库根目录，使输出路径不依赖当前工作目录。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
csv_path="${repo_root}/results/raw/reduction_v2_comparison.csv"

# 覆盖 Warp、Block、非 2 的幂、非整除和大规模多阶段归约。
sizes=(
    1
    31
    32
    33
    255
    256
    257
    511
    512
    513
    1023
    1024
    1025
    1000003
    16777219
)

# 对每个 N 依次测 V0/V1/V2；输入、预热和统计语义保持完全一致。
kernels=(interleaved sequential first_add)
for input_count in "${sizes[@]}"; do
    for kernel in "${kernels[@]}"; do
        "${repo_root}/build/reduction_bench" \
        --kernel "${kernel}" \
        --size "${input_count}" \
        --warmup 20 \
        --iterations 100 \
        --groups 5 \
        --csv "${csv_path}"
    done
done
