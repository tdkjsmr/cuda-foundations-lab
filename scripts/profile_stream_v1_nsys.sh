#!/usr/bin/env bash

# NSYS 采集或任何统计导出失败都立即退出。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nsys_bin="${NSYS_BIN:-/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys}"
report_dir="${repo_root}/results/nsys"
report_base="${report_dir}/stream_v1_pinned_sync"
mkdir -p "${report_dir}"

# 与 V3.0 使用同一 32 MiB × 16 Chunks；预热位于 capture-range 之外。
"${nsys_bin}" profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --stats=false \
    --force-overwrite=true \
    --output "${report_base}" \
    "${repo_root}/build/stream_bench" \
    --mode pinned_sync \
    --shape 4096x2048 \
    --chunks 16 \
    --streams 1 \
    --warmup 1 \
    --profile

# 终端先展示可直接回传的 Host API、GPU 活动、逐事件和 NVTX 汇总。
"${nsys_bin}" stats \
    --report cuda_api_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --report cuda_gpu_trace \
    --report nvtx_pushpop_sum \
    --report nvtx_gpu_proj_sum \
    --format column \
    --force-export=true \
    --output - \
    "${report_base}.nsys-rep"

# 同时导出 CSV，后续与 stream_v0_pageable_sync 的相同字段逐项比较。
"${nsys_bin}" stats \
    --report cuda_api_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --report cuda_gpu_trace \
    --report nvtx_pushpop_sum \
    --report nvtx_gpu_proj_sum \
    --format csv \
    --output "${report_base}" \
    --force-export=true \
    --force-overwrite=true \
    "${report_base}.nsys-rep"
