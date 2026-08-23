#!/usr/bin/env bash

# 任意一组 Stream Benchmark 失败时立即退出。
set -euo pipefail

# 输出路径始终相对于仓库根目录，不依赖调用时的工作目录。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="${repo_root}/results/raw"
mkdir -p "${results_dir}"

csv_path="${results_dir}/stream_v0_pageable_sync.csv"
# 只有全部 3 组正式实验完成后才原子替换旧数据。
benchmark_csv_tmp="$(mktemp "${csv_path}.tmp.XXXXXX")"
trap 'rm -f "${benchmark_csv_tmp}"' EXIT

# 三种 Chunk 都保持整批输入 Payload = 512 MiB，只改变单 Chunk 粒度和任务数。
shapes=(4096x512 4096x2048 4096x4096)
chunks=(64 16 8)

for index in "${!shapes[@]}"; do
    shape="${shapes[index]}"
    width="${shape%x*}"
    height="${shape#*x}"
    payload_bytes="$((width * height * 4 * chunks[index]))"
    expected_payload_bytes="$((512 * 1024 * 1024))"
    if [[ "${payload_bytes}" -ne "${expected_payload_bytes}" ]]; then
        echo "Stream 工作量不是固定 512 MiB：${shape} × ${chunks[index]} chunks" >&2
        exit 1
    fi

    "${repo_root}/build/stream_bench" \
        --mode pageable_sync \
        --shape "${shapes[index]}" \
        --chunks "${chunks[index]}" \
        --streams 1 \
        --warmup 20 \
        --iterations 100 \
        --groups 5 \
        --csv "${benchmark_csv_tmp}"
done

# 正式文件必须恰好包含 1 行表头和 3 行结果。
benchmark_csv_lines="$(wc -l < "${benchmark_csv_tmp}")"
if [[ "${benchmark_csv_lines}" -ne 4 ]]; then
    echo "Stream V3.0 CSV 行数异常：期望 4，实际 ${benchmark_csv_lines}" >&2
    exit 1
fi

# mktemp 产生私有权限，发布前恢复仓库结果文件的常规权限。
chmod 0644 "${benchmark_csv_tmp}"
mv "${benchmark_csv_tmp}" "${csv_path}"
