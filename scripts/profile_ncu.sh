#!/usr/bin/env bash

# Profiler 或目标程序失败时立即退出，不保留看似成功的不完整报告。
set -euo pipefail

# 从脚本位置解析仓库根目录，避免报告输出到未知工作目录。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 优先使用 PATH 中的 ncu；当前服务器通常位于 /usr/local/cuda/bin/ncu。
ncu_bin="${NCU_BIN:-/usr/local/cuda/bin/ncu}"

# 第一个参数选择待分析版本；默认针对 v1.2 新增的 tiled。
kernel_name="${1:-tiled}"
if [[ "${kernel_name}" != "copy" && "${kernel_name}" != "naive" &&
      "${kernel_name}" != "tiled" ]]; then
    echo "用法: $0 [copy|naive|tiled]" >&2
    exit 1
fi

# 报告名称绑定当前 Transpose 阶段和 Kernel，避免覆盖不同版本的证据。
report_base="${repo_root}/results/ncu/transpose_v2_${kernel_name}"

# 程序先执行 20 次 Warm-up；NCU 跳过它们，只采集随后唯一一次正式 Kernel。
"${ncu_bin}" \
    --launch-skip 20 \
    --launch-count 1 \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --section WarpStateStats \
    --import-source yes \
    --export "${report_base}" \
    --force-overwrite \
    "${repo_root}/build/transpose_bench" \
    --kernel "${kernel_name}" \
    --shape 4096x4096 \
    --warmup 20 \
    --profile

# 把报告的详细页面同时导出为纯文本，便于无 GUI 环境审阅和回传。
"${ncu_bin}" \
    --import "${report_base}.ncu-rep" \
    --page details \
    --log-file "${report_base}_details.txt"

# 把全部原始 Metric 导出为 CSV，便于后续版本做同指标比较。
"${ncu_bin}" \
    --import "${report_base}.ncu-rep" \
    --page raw \
    --csv \
    --log-file "${report_base}_raw.csv"
