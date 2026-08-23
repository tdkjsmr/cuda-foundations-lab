#!/usr/bin/env bash

# Nsight Systems 或目标程序失败时立即退出。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nsys_bin="${NSYS_BIN:-/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys}"
report_base="${repo_root}/results/nsys/reduction_v1_sequential"

# N=1,000,003 的 Sequential 每次完整归约包含 3 个阶段；5 次预热后只测 1 次正式归约。
"${nsys_bin}" profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --stats=false \
    --force-overwrite=true \
    --output "${report_base}" \
    "${repo_root}/build/reduction_bench" \
    --kernel sequential \
    --size 1000003 \
    --warmup 5 \
    --profile

# 终端优先展示 CUDA API、各阶段 Kernel 汇总和逐事件 GPU 时间线。
"${nsys_bin}" stats \
    --report cuda_api_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --report cuda_gpu_trace \
    --format column \
    --force-export=true \
    --output - \
    "${report_base}.nsys-rep"

# 同一组报告额外导出 CSV，便于后续版本逐字段比较。
"${nsys_bin}" stats \
    --report cuda_api_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --report cuda_gpu_trace \
    --format csv \
    --output "${report_base}" \
    --force-export=true \
    --force-overwrite=true \
    "${report_base}.nsys-rep"
