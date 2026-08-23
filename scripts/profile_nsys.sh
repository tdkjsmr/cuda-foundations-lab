#!/usr/bin/env bash

# Nsight Systems 或目标程序失败时立即停止。
set -euo pipefail

# 从脚本位置解析仓库根目录，保证报告落在当前项目中。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 当前服务器的 nsys 未加入 PATH，因此默认使用已验证的绝对路径。
nsys_bin="${NSYS_BIN:-/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys}"

# 第一个位置参数选择 copy 或 naive；默认分析本版本新增的 naive。
kernel_name="${1:-naive}"
if [[ "${kernel_name}" != "copy" && "${kernel_name}" != "naive" ]]; then
    echo "用法: $0 [copy|naive]" >&2
    exit 1
fi

# 不启用 GPU Metrics Sampling，只采集当前容器已验证可用的 CUDA/NVTX/OSRT 时间线。
report_base="${repo_root}/results/nsys/transpose_v1_${kernel_name}"
"${nsys_bin}" profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --stats=true \
    --force-overwrite=true \
    --output "${report_base}" \
    "${repo_root}/build/transpose_bench" \
    --kernel "${kernel_name}" \
    --shape 4096x4096 \
    --warmup 5 \
    --profile

# 输出最有用的 API、Kernel、显存汇总和逐事件 GPU 时间线到终端。
"${nsys_bin}" stats \
    --report cuda_api_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --report cuda_gpu_trace \
    --format column \
    --output - \
    "${report_base}.nsys-rep"

# 额外导出 CSV，便于无 GUI 环境做版本间对照。
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
