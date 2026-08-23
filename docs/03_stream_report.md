# Stream 子项目报告：V0 Pageable + Synchronous

## 1. 阶段状态

本文档对应 Git 分支 `v3.0`，也就是 Stream 子项目的第一轮：

```text
Pageable Host Memory
→ blocking cudaMemcpy H2D
→ Padded Tiled Transpose
→ blocking cudaMemcpy D2H
```

这一轮的目标不是进行重叠，而是建立一条可重现、可验证、可被 Nsight Systems 解释的串行基线。后续两轮将依次只引入：

1. `v3.1`：Pinned Host Memory + Synchronous；
2. `v3.2`：Pinned Host Memory + `cudaMemcpyAsync()` + 多个非默认 Stream。

因此，V3.0 时间线“没有重叠”是成功验收，而不是优化失败。

## 2. 冻结的实验合同

### 2.1 工作负载

一个 Chunk 就是一张独立的 FP32 矩阵。每个 Chunk 使用完全相同的逻辑流程：

```text
Host Input Chunk
→ Device Input Buffer
→ stream_padded_transpose_kernel
→ Device Output Buffer
→ Host Output Chunk
```

正式实验把整批输入 Payload 固定为 `512 MiB`，仅改变单 Chunk 粒度和 Chunk 数：

| Chunk | 单矩阵 Shape | Chunk 数 | 总输入 Payload |
|---:|---:|---:|---:|
| 8 MiB | `4096 × 512` | 64 | 512 MiB |
| 32 MiB | `4096 × 2048` | 16 | 512 MiB |
| 64 MiB | `4096 × 4096` | 8 | 512 MiB |

这种设计保证：

- 不同 Chunk 粒度之间处理的总字节一致；
- 64 MiB 配置仍有 8 个独立任务，后续可以完整测试 8 个 Stream；
- 每个 Chunk 都是完整矩阵，没有“从矩阵中间切开后如何布置转置输出”的额外语义。

`scripts/run_stream_v0_benchmarks.sh` 在运行每组实验前会重新计算 Payload，不是 512 MiB 就立即失败。

### 2.2 冻结的 Kernel

Stream 子项目复用 Transpose 最终版的算法：

- `32 × 8 = 256` Threads/Block；
- 每个 Block 处理一个 `32 × 32` Tile；
- Shared Memory 为 `float tile[32][33]`；
- Global Load 和 Global Store 都按 Warp 连续；
- 非 32 整除 Shape 在 Load/Store 两端都有独立边界检查。

为满足本项目“每个 `src/<module>/*.cu` 能独立从 Kernel 读到 Host 控制流和 `main()`”的学习要求，`src/streams/stream_pipeline.cu` 保留了这一已验证算法的冻结副本。后续 Stream 版本不得修改 Kernel、Grid、Block 或输入内容，否则无法把性能差异归因于 Host Memory 和提交方式。

## 3. 代码总体框架

```text
main / stream_test / stream_bench
        │
        ├─ validate_and_derive_layout
        ├─ query_device_capabilities
        ├─ allocate_device_buffers
        │
        └─ execute_synchronous_pipeline
              └─ for each chunk
                    ├─ cudaMemcpy H2D
                    ├─ launch_padded_tiled_transpose
                    └─ cudaMemcpy D2H
```

文件职责：

| 文件 | 职责 |
|---|---|
| `include/stream_pipeline.cuh` | Workload/Layout、Device 能力、Move-only Buffer 所有权和公共接口 |
| `src/streams/stream_pipeline.cu` | Padded Kernel、Launcher、同步 Pipeline、容量检查、NVTX 和演示 `main()` |
| `tests/stream_test.cu` | CPU Reference、边界、多 Chunk、特殊位模式、所有权和溢出测试 |
| `benchmarks/stream_bench.cu` | CPU Submit、GPU Span、End-to-end 统计、CSV 和 Profile 模式 |

### 3.1 V3.0 的严格提交顺序

```cpp
cudaMemcpy(..., cudaMemcpyHostToDevice);
launch_padded_tiled_transpose(..., nullptr);
cudaMemcpy(..., cudaMemcpyDeviceToHost);
```

重要性质：

- Host Input/Output 由 `std::vector<float>` 分配，属于 Pageable Memory；
- 传输 API 只使用 blocking `cudaMemcpy()`；
- Kernel 显式传入 `nullptr` Stream，即默认 Stream；
- Chunk 循环内没有 `cudaDeviceSynchronize()`；
- blocking D2H 返回时，当前 Chunk 的 Kernel 已完成，所以下一 Chunk 才可安全复用同一对 Device Buffer；
- 整个 Batch 边界再调用 `cudaDeviceSynchronize()` 捕获异步执行错误。

### 3.2 资源与错误处理

- 所有 CUDA Runtime API 都经过 `CUDA_CHECK`；
- Kernel Launch 后立即调用 `cudaGetLastError()`；
- Shape、单 Chunk 元素数、FP32 字节数和整批 Chunk 数都有 `size_t` 溢出检查；
- Grid X/Y 超出当前 `sm_86` 目标范围会被拒绝；
- 第二次 `cudaMalloc()` 失败时先回收第一块 Buffer；
- `DeviceBufferPair` 禁止复制，只允许 Move，析构也会兜底回收；
- 正常路径仍显式按 Output → Input 逆序释放。

## 4. 正确性与安全性

### 4.1 CPU Reference

对于每个 Chunk，CPU 都独立执行：

```text
input[chunk_offset + y * width + x]
→ output[chunk_offset + x * height + y]
```

输入模式显式融入 Chunk ID，因此“每次都错误处理 Chunk 0”不会误通过。Transpose 只重排数据，不改变数值，所以验证使用 IEEE-754 逐位比较，不使用数值误差容差。

### 4.2 常规测试覆盖

- `1 × 1`，1 Chunk；
- `31 × 33`，小于/跨越 Tile，3 Chunks；
- `32 × 32`，恰好一个 Tile；
- `33 × 31`，非整除矩形，5 Chunks；
- `257 × 129`，9 Chunks，为未来 8-Stream 不整除任务数回归；
- `4096 × 512`，2 个 8 MiB Chunk Smoke Test；
- `+0/-0`、Subnormal、最大有限值、正负无穷和两种 NaN payload；
- width/height/chunk 为 0、元素/字节/Batch 溢出、Grid 越界、Layout 不一致、零 Device 容量；
- Move 后的 Device Buffer 源/目标所有权状态。

实际验证结果：

```text
CTest: 3/3 PASS
Stream test: PASS
Compute Sanitizer memcheck: 0 errors
Compute Sanitizer leak-check: 0 bytes leaked
```

## 5. 三种计时口径

统一规范要求 Stream 同时测三类时间。本实现从 V3.0 就冻结了 CSV Schema。

### 5.1 CPU Submit Time

```text
steady_clock start
→ Host 调用完整批 H2D / Kernel / D2H
→ steady_clock submit stop
```

在 V3.0/V3.1 中，blocking `cudaMemcpy()` 会让 CPU Submit Time 包含大量等待，因此它会非常接近 End-to-end。这不是测量错误，而是同步基线的本质。

### 5.2 GPU Span Time

两个 CUDA Event 在默认 Stream 上包围完整批 GPU 任务。该指标包含 GPU 时间线上的间隙，不等于 Kernel/Copy Engine Busy Time 之和。V3.2 多 Stream 版将使用控制 Stream + Worker Completion Events 建立全局包络。

### 5.3 End-to-end Time

```text
cudaDeviceSynchronize
→ steady_clock start
→ 提交完整批任务
→ cudaDeviceSynchronize
→ steady_clock stop
```

这是 Stream 子项目的主要性能指标，包含 `H2D + Kernel + D2H`。分配、输入生成、CPU Reference 和验证都在计时区间外。

### 5.4 吞吐率定义

```text
Payload Throughput = 原始输入 Payload / End-to-end P50
```

这个数字表示应用每秒完成多少原始输入数据。

报告另外保留：

```text
Accounted Traffic Rate = 4 × Payload / End-to-end P50
```

`4B` 分别来自 H2D、Kernel Read、Kernel Write 和 D2H。它跨越 PCIe 与 GPU DRAM 两种不同 Fabric，不能与某一个硬件峰值带宽直接比较。

## 6. 正式 Benchmark 结果

实验条件：

- Source commit: `60143eb`；
- GPU: NVIDIA GeForce RTX 3090；
- CUDA Driver API: 13.2；
- CUDA Runtime: 12.4；
- Warm-up: 20 次完整 Pipeline；
- 正式测量: 100 次/组，5 组；
- 统计对象: 5 个“单次完整 Pipeline 平均时间”；
- 原始数据: `results/raw/stream_v0_pageable_sync.csv`。

### 6.1 三类时间中位数

| Chunk | CPU Submit P50 | GPU Span P50 | End-to-end P50 |
|---:|---:|---:|---:|
| 8 MiB | 117.085 ms | 117.088 ms | 117.086 ms |
| 32 MiB | 113.447 ms | 113.444 ms | 113.448 ms |
| 64 MiB | 111.084 ms | 111.078 ms | 111.084 ms |

三个数在同步基线中接近，与 Host 大部分时间阻塞在 `cudaMemcpy()` 的同步语义一致；该语义同时由源码中的 blocking API 与 NSYS `cuda_api_sum`/时间线确认。

### 6.2 End-to-end 稳定态统计

| Chunk | Min | P50 | P95 | Stddev | Payload Throughput | 4B/T |
|---:|---:|---:|---:|---:|---:|---:|
| 8 MiB | 116.058 ms | 117.086 ms | 124.524 ms | 3.695 ms | 4.585 GB/s | 18.341 GB/s |
| 32 MiB | 112.567 ms | 113.448 ms | 121.711 ms | 4.023 ms | 4.732 GB/s | 18.929 GB/s |
| 64 MiB | 110.784 ms | 111.084 ms | 118.048 ms | 3.171 ms | 4.833 GB/s | 19.332 GB/s |

相对关系：

- 32 MiB 相对 8 MiB：`1.0321×`；
- 64 MiB 相对 8 MiB：`1.0540×`；
- 64 MiB 相对 32 MiB：`1.0213×`；
- End-to-end 变异系数分别为 `3.16% / 3.55% / 2.85%`。

在总 Payload 相同时，更大 Chunk 减少了 API 调用和 Kernel Launch 数，因此 V3.0 串行基线中 64 MiB 最快。但这不能推导“未来 Stream 版 Chunk 越大越好”：进入异步 Pipeline 后，过大 Chunk 会降低任务粒度，可能无法充分填满不同引擎。

`speedup_vs_pinned_sync` 在 V3.0 CSV 中为 `nan`，因为 Pinned Sync 还没有实测。不用 0 伪装未存在的基线。

## 7. Nsight Systems 证据

### 7.1 采集命令

```bash
./scripts/profile_stream_v0_nsys.sh
```

脚本使用 CUDA Profiler API capture-range，只保留一次 `32 MiB × 16 chunks = 512 MiB` 正式 Pipeline；NVTX 范围保留 `profile`、`pipeline_iteration`、`chunk_N`、`h2d`、`transpose` 和 `d2h`。

输出包含：

- `results/nsys/stream_v0_pageable_sync.nsys-rep`；
- `cuda_api_sum`；
- `cuda_gpu_kern_sum`；
- `cuda_gpu_mem_time_sum`；
- `cuda_gpu_trace`；
- `nvtx_pushpop_sum`；
- `nvtx_gpu_proj_sum`。

### 7.2 优先分析顺序

#### 第一步：`cuda_gpu_trace`

这是判断“是否重叠”的主证据。本报告中：

- 16 次 H2D；
- 16 次 Kernel；
- 16 次 D2H；
- 共48个 GPU Operations；
- 全部位于同一个默认 Stream（NSYS 内部 Stream ID 为 7）；
- 序列严格重复 `H2D → Kernel → D2H`；
- 不存在任何跨 Chunk 时间重叠。

这与 V3.0 的源码合同完全一致。

#### 第二步：`cuda_gpu_mem_time_sum`

| 阶段 | 调用数 | 总时间 | 平均 | 中位数 |
|---|---:|---:|---:|---:|
| H2D | 16 | 52.498 ms | 3.281 ms | 3.273 ms |
| D2H | 16 | 63.239 ms | 3.952 ms | 3.445 ms |

D2H 的一次 8.955 ms 长尾使平均值高于中位数。NSYS 本身会扰动单次轨迹，因此性能排名必须使用 20 预热、100 × 5 的正式 CSV，不用这一次 Profiler 时间代替 Benchmark。

#### 第三步：`cuda_gpu_kern_sum`

```text
Instances: 16
Total: 1.298 ms
Average: 81.125 us
Median: 81.094 us
Registers/thread: 26
Static Shared Memory: 4,224 bytes
Grid/chunk: (128, 64, 1)
Block: (32, 8, 1)
```

#### 第四步：三阶段占比

GPU Operation 活跃时间记账合计：

```text
H2D    52.498 ms   44.86%
Kernel  1.298 ms    1.11%
D2H    63.239 ms   54.03%
Total 117.035 ms
```

NVTX GPU Projection 的整体 Span 为 `121.936 ms`，比三阶段活跃时间之和多约 `4.901 ms`，对应 API 提交/调度间隙。时间线证明三阶段串行，所以这里不存在因重叠导致的“活跃和大于 Span”。

#### 第五步：`cuda_api_sum`

```text
cudaMemcpy:       32 calls, 121.943 ms, 99.7% Host API time
cudaLaunchKernel: 16 calls,   0.230 ms,  0.2% Host API time
```

Pageable + blocking 基线的 Host Thread 几乎全程卡在 `cudaMemcpy()` 中，所以 CPU Submit 时间和 End-to-end 时间接近是必然结果。

### 7.3 硬件能力边界

程序实测输出：

```text
deviceOverlap: 1
asyncEngineCount: 2
concurrentKernels: 1
```

这些字段表明设备具备潜在的传输/计算并发能力。它们不能证明本程序已经重叠；V3.0 的 NSYS 时间线实际证明完全串行。

## 8. 构建与复现

```bash
export PATH=/root/.local/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j

./build/stream_pipeline --shape 31x33 --chunks 3
./build/stream_test
./build/stream_bench \
  --mode pageable_sync \
  --shape 4096x2048 \
  --chunks 16 \
  --streams 1 \
  --warmup 20 \
  --iterations 100 \
  --groups 5

./scripts/run_stream_v0_benchmarks.sh
./scripts/profile_stream_v0_nsys.sh
```

Sanitizer：

```bash
compute-sanitizer \
  --tool memcheck \
  --leak-check full \
  --error-exitcode 1 \
  ./build/stream_test
```

## 9. V3.0 阶段性验收问题

### 1. 同一 Stream 内的任务为什么天然有序？

同一 Stream 表示一条有序的 Device 工作队列；后提交的操作在该 Stream 内不会越过前面的操作执行。

### 2. 不同 Stream 一定并发吗？

不一定。Stream 只允许并发，实际执行还受硬件引擎、资源占用、数据依赖、默认 Stream 语义和任务粒度影响。

### 3. `cudaMemcpyAsync()` 为什么通常需要 Pinned Memory？

DMA 需要在传输期间保持稳定、不被 OS 换出的 Host 物理页。Pageable Memory 往往需要 Runtime 额外暂存到内部 Pinned Buffer，并可能使 API 变成阻塞或无法达到真正异步。

### 4. Pinned Memory 为什么不能无限使用？

Pinned 页不能被 OS 正常换出，过多使用会减少系统可分页内存、增加内存压力，并可能使整机性能恶化。

### 5. 默认 Stream 为什么可能引入隐式同步？

Legacy Default Stream 与其他 Stream 之间可能存在全局排序语义。并发 Pipeline 应显式使用非默认 Stream，避免无意间把本可并发的工作串行化。

### 6. `asyncEngineCount` 说明什么？

它描述设备可用的异步 Copy Engine 能力。数值为 2 通常有利于同时处理两个方向的传输，但不保证任意代码或任意传输组合一定重叠。

### 7. 为什么 8 个 Stream 可能比 4 个更慢？

硬件引擎数量有限，额外 Stream 会带来更多 API 提交、排队、Event/Buffer 管理和资源竞争；当 4 个 Stream 已填满可用引擎时，再增加 Stream 不会创造更多硬件并行能力。

### 8. 为什么 Stream 实验必须使用 Nsight Systems？

总时间变少可能来自拖频减轻、缓存、抖动或较少 API 开销，不能证明 H2D/Kernel/D2H 已重叠。只有 NSYS 时间线可以直接展示不同 Stream 上各操作的开始、结束和重叠区间。

## 10. 证据边界与下一轮

NCU 在当前 AutoDL Docker 中因 `ERR_NVGPUCTRPERM` 无法访问硬件计数器。Stream 子项目要回答的是程序级时间线重叠，因此关键验收工具本来就是 NSYS，不会用 NSYS 数据冒充 NCU 硬件计数器。

V3.1 将保持以下项全部不变：

- 三种 Shape/Chunk 数和 512 MiB 总 Payload；
- Padded Tiled Kernel、Grid 和 Block；
- blocking `cudaMemcpy()`；
- 默认 Stream 和串行提交顺序；
- 正确性、三类计时、统计和 CSV Schema。

唯一主要实验变量将是：

```text
std::vector Pageable Allocation
→ cudaMallocHost Pinned Allocation
```

这样才能回答“只替换 Host Memory 类型，同步传输本身变化多少”。
