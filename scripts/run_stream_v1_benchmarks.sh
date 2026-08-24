#!/usr/bin/env bash

# 任意一组 Pinned Sync Benchmark 失败时立即退出，避免发布不完整数据。
set -euo pipefail

# 所有路径都从脚本位置推导，不依赖用户当前所在目录。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="${repo_root}/results/raw"
mkdir -p "${results_dir}"

csv_path="${results_dir}/stream_v1_pinned_sync.csv"
# 三组全部成功后再原子替换正式 CSV；失败时 EXIT trap 删除临时文件。
benchmark_csv_tmp="$(mktemp "${csv_path}.tmp.XXXXXX")"
trap 'rm -f "${benchmark_csv_tmp}"' EXIT

# 与 V3.0 完全相同：三种 Chunk 都处理固定 512 MiB 原始输入 Payload。
shapes=(4096x512 4096x2048 4096x4096)
chunks=(64 16 8)
expected_payload_bytes="$((512 * 1024 * 1024))"

for index in "${!shapes[@]}"; do
    shape="${shapes[index]}"
    width="${shape%x*}"
    height="${shape#*x}"
    payload_bytes="$((width * height * 4 * chunks[index]))"

    # Workload 变化会破坏与 Pageable Sync 的单变量对比，因此立即拒绝。
    if [[ "${payload_bytes}" -ne "${expected_payload_bytes}" ]]; then
        echo "Stream V3.1 工作量不是固定 512 MiB：${shape} × ${chunks[index]} chunks" >&2
        exit 1
    fi

    "${repo_root}/build/stream_bench" \
        --mode pinned_sync \
        --shape "${shape}" \
        --chunks "${chunks[index]}" \
        --streams 1 \
        --warmup 20 \
        --iterations 100 \
        --groups 5 \
        --csv "${benchmark_csv_tmp}"
done

# 正式文件必须恰好包含一行 Header 和三行 Pinned Sync 结果。
benchmark_csv_lines="$(wc -l < "${benchmark_csv_tmp}")"
if [[ "${benchmark_csv_lines}" -ne 4 ]]; then
    echo "Stream V3.1 CSV 行数异常：期望 4，实际 ${benchmark_csv_lines}" >&2
    exit 1
fi

# mktemp 默认权限较严格；发布到仓库前恢复普通只读数据文件权限。
chmod 0644 "${benchmark_csv_tmp}"
mv "${benchmark_csv_tmp}" "${csv_path}"
