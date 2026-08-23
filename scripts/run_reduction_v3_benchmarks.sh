#!/usr/bin/env bash

# 任意 Benchmark 失败时立即停止，避免保留不完整的正式数据。
set -euo pipefail

# 从脚本位置解析仓库根目录，使输出路径不依赖当前工作目录。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
csv_path="${repo_root}/results/raw/reduction_v3_comparison.csv"
# 先写入同目录临时文件；只有 60 条实验全部成功后才原子替换正式 CSV。
benchmark_csv_tmp="$(mktemp "${csv_path}.tmp.XXXXXX")"
# 脚本失败或被中断时移除不完整临时数据，同时保留上一份正式结果。
trap 'rm -f "${benchmark_csv_tmp}"' EXIT

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

# 对每个 N 依次测 V0/V1/V2/V3；输入、预热和统计语义保持完全一致。
kernels=(interleaved sequential first_add warp_shuffle)
for input_count in "${sizes[@]}"; do
    for kernel in "${kernels[@]}"; do
        "${repo_root}/build/reduction_bench" \
        --kernel "${kernel}" \
        --size "${input_count}" \
        --warmup 20 \
        --iterations 100 \
        --groups 5 \
        --csv "${benchmark_csv_tmp}"
    done
done

# 正式文件应包含 1 行表头和 15 × 4 行结果；数量异常时拒绝发布。
benchmark_csv_lines="$(wc -l < "${benchmark_csv_tmp}")"
if [[ "${benchmark_csv_lines}" -ne 61 ]]; then
    echo "Reduction V3 CSV 行数异常：期望 61，实际 ${benchmark_csv_lines}" >&2
    exit 1
fi

# 全部四版本、15 个 N 完成后再发布新结果，使脚本可安全重复执行。
# 临时文件由 mktemp 创建为私有权限，发布前恢复常规 0644 结果文件权限。
chmod 0644 "${benchmark_csv_tmp}"
mv "${benchmark_csv_tmp}" "${csv_path}"
