#!/usr/bin/env bash

# 当前容器会返回 ERR_NVGPUCTRPERM；保留脚本供允许计数器的环境直接运行。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ncu_bin="${NCU_BIN:-/usr/local/cuda/bin/ncu}"
report_base="${repo_root}/results/ncu/reduction_v3_warp_shuffle"

# N=1,000,003 的 Warp Shuffle 每次有 3 个阶段（Grid 1954 → 4 → 1）；跳过 5×3 个预热 Launch，只采集正式第一阶段。
"${ncu_bin}" \
    --launch-skip 15 \
    --launch-count 1 \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --section WarpStateStats \
    --import-source yes \
    --export "${report_base}" \
    --force-overwrite \
    "${repo_root}/build/reduction_bench" \
    --kernel warp_shuffle \
    --size 1000003 \
    --warmup 5 \
    --profile

"${ncu_bin}" \
    --import "${report_base}.ncu-rep" \
    --page details \
    --log-file "${report_base}_details.txt"

"${ncu_bin}" \
    --import "${report_base}.ncu-rep" \
    --page raw \
    --csv \
    --log-file "${report_base}_raw.csv"
