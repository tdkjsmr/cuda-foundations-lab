# CUDA Foundations Lab

这是 OLCF CUDA Training Series 阶段收尾项目，目标是建立“正确性、稳定测量、Profiler 证据、单变量迭代”的 CUDA 性能工程闭环。

当前开发分支：`v3.0`。

Transpose 子项目最终分支：`v1.3`。

Reduction 子项目最终分支：`v2.3`。

Stream 子项目当前分支：`v3.0`（Pageable + Synchronous 基线）。

当前保留四个 Transpose 版本：

- V0 Copy Baseline：`output[y][x] = input[y][x]`；
- V1 Naive Transpose：`output[x][y] = input[y][x]`；
- V2 Shared Memory Tiled：使用未 Padding 的 `tile[32][32]` 重排数据；
- V3 Padded Shared Memory：使用 `tile[32][33]` 改变 Shared Memory Bank 映射。

V0 建立连续读写的实际带宽参考。V1 把连续输出写入改为跨步写入。V2 通过 Shared Memory Tiling 恢复连续 Global Load 和 Store，但保留未 Padding Tile 的 Bank Conflict。V3 仅增加一列 Padding，保持其余数据路径不变。

## 目录关系

```text
include/                    CUDA 错误检查、Event 计时、统计和公共接口
src/transpose/transpose.cu  V0/V1/V2/V3 Kernel、Host 启动逻辑和演示 main
src/reduction/reduction.cu  V0/V1/V2/V3 Kernel、GPU 多阶段控制和演示 main
src/streams/stream_pipeline.cu  Padded Transpose Kernel、同步 Pipeline 和演示 main
tests/transpose_test.cu     CPU Reference、正确性、边界和特殊位模式测试
tests/reduction_test.cu     规定 N、数据分布、误差与多阶段测试
tests/stream_test.cu        多 Chunk、边界、位模式、溢出和所有权测试
benchmarks/                 Transpose/Reduction Kernel-only 与 Stream E2E Benchmark
scripts/                    测试、Benchmark、NCU/NSYS 命令
results/                    原始 CSV 与 Profiler 报告
docs/                       中文实现与性能分析报告
```

## Transpose 最终状态

- 四个版本：全部保留并可通过 `--kernel` 选择；
- 正确性：全部规定 Shape、非整除边界和特殊位模式逐位通过；
- 安全性：`memcheck` 0 errors，`racecheck` 0 hazards/errors/warnings；
- 性能：Padded 相对 Naive 加速 `2.72×–3.15×`，达到 Copy 的 `93.68%–100.31%`；
- Profiler：V0–V3 NSYS 报告已归档；NCU 因容器权限不可采集。

## Reduction 最终状态（V0–V3）

- Interleaved Addressing：已保留为 V0 基线；
- Sequential Addressing：已实现为 V1，连续前半线程参与归约；
- First Add During Load：已实现为 V2，每线程最多先合并两个输入；
- Warp Shuffle：已实现为 V3，最后 64 个 Partial 由首 Warp 在 Register 中完成；
- 任意 N 和非 2 的幂：已支持；
- GPU 多阶段归约：已支持，两块 Workspace Ping-Pong 到单个结果；
- CPU Reference：double 串行累加；
- 误差：absolute error、normalized error 和输入相关 tolerance；
- 测试：四版本 156 个用例，覆盖规定 N、五种分布、Shuffle/多阶段边界和 One-Hot 定位；
- Benchmark：V3 在 N=1,000,003 和 16,777,219 上分别较 V2 加速 1.394× 和 1.382×；
- NSYS：V3 保持 `1954 → 4 → 1`，三阶段均加速，18 次 Kernel 总时间较 V2 提升 1.426×；NCU 仍受容器权限限制。

## Stream 当前状态（V0）

- 工作负载：复用 Padded Tiled Transpose，完整覆盖 H2D + Kernel + D2H；
- Host Memory：普通 `std::vector` Pageable Memory；
- 提交方式：blocking `cudaMemcpy` + 默认 Stream，严格串行；
- 公平性：8/32/64 MiB Chunk 均固定 512 MiB 总输入 Payload；
- 测量：同时记录 CPU Submit、CUDA Event GPU Span 和 steady-clock End-to-end；
- 正确性：多 Chunk 与特殊 IEEE-754 位模式逐位通过；
- 安全性：`memcheck` 0 errors，`leak-check` 0 bytes；
- NSYS：16 组 H2D → Kernel → D2H 完全串行，作为后续重叠实验的时间线基线。

## 配置、构建和运行

```bash
export PATH=/root/.local/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j

./build/transpose --kernel copy --shape 31x33
./build/transpose --kernel naive --shape 31x33
./build/transpose --kernel tiled --shape 31x33
./build/transpose --kernel padded --shape 31x33

./scripts/run_tests.sh
./scripts/run_benchmarks.sh
./scripts/profile_nsys.sh tiled
./scripts/profile_nsys.sh padded

./build/reduction --kernel warp_shuffle --size 1000003 --pattern random
./scripts/run_reduction_v3_benchmarks.sh
./scripts/profile_reduction_v3_nsys.sh

./build/stream_pipeline --shape 31x33 --chunks 3
./build/stream_bench --mode pageable_sync --shape 4096x2048 --chunks 16 --streams 1
./scripts/run_stream_v0_benchmarks.sh
./scripts/profile_stream_v0_nsys.sh
```

## 版本演进

| 分支 | 实现 | 主要实验变量 |
|---|---|---|
| `v1.0` | Copy | 连续读写带宽基线 |
| `v1.1` | Naive | 跨步 Global Store |
| `v1.2` | Tiled | `tile[32][32]` 恢复连续 Global Store |
| `v1.3` | Padded | `tile[32][33]` 改变 Shared Memory Bank 映射 |
| `v2.0` | Interleaved Reduction | 交错活跃线程与完整 GPU 多阶段归约基线 |
| `v2.1` | Sequential Reduction | 连续前半线程归约，减少 Warp 分支浪费 |
| `v2.2` | First Add During Load | 每线程合并两个输入，减少 Block 和 Partial Sums |
| `v2.3` | Warp Shuffle Reduction | 用 Register Shuffle 完成最后一个 Warp，减少 Shared 访问与 Block 屏障 |
| `v3.0` | Pageable Synchronous Stream Baseline | 普通 Host Memory + blocking H2D/Kernel/D2H 串行基线 |

## 当前交付物

- [统一源码](src/transpose/transpose.cu)：四个 Kernel、Host Launcher 与演示 `main()`；
- [正确性测试](tests/transpose_test.cu)：CPU Reference、全部 Shape 和特殊位模式；
- [稳定 Benchmark](benchmarks/transpose_bench.cu)：CUDA Event、统计量和 CSV；
- [完整性能报告](docs/01_transpose_report.md)：实现原理、结果、NSYS 证据与口头验收答案；
- [最终原始数据](results/raw/transpose_v3.csv)：四版本、四种正式 Shape；
- `results/nsys/`：各阶段 `.nsys-rep` 与命令行 CSV 摘要。

Reduction V0/V1/V2/V3：

- [统一源码](src/reduction/reduction.cu)：Interleaved/Sequential/First Add/Warp Shuffle Kernel、GPU 多阶段 Host 控制与演示 `main()`；
- [正确性测试](tests/reduction_test.cu)：全部规定 N、五类输入和误差验收；
- [稳定 Benchmark](benchmarks/reduction_bench.cu)：完整多阶段 CUDA Event 计时；
- [性能报告](docs/02_reduction_report.md)：算法、测试、正式数据与 NSYS 阶段分析；
- [V0 原始数据](results/raw/reduction_v0.csv) 与 `results/nsys/reduction_v0_interleaved*`。
- [V1 对比数据](results/raw/reduction_v1_comparison.csv) 与 `results/nsys/reduction_v1_sequential*`。
- [V2 对比数据](results/raw/reduction_v2_comparison.csv) 与 `results/nsys/reduction_v2_first_add*`。
- [V3 四版本对比数据](results/raw/reduction_v3_comparison.csv) 与 `results/nsys/reduction_v3_warp_shuffle*`。

Stream V0：

- [统一源码](src/streams/stream_pipeline.cu)：Padded Kernel、同步 Host Pipeline、NVTX 和演示 `main()`；
- [正确性测试](tests/stream_test.cu)：多 Chunk、非整除 Shape、特殊位模式、溢出和 Move-only 所有权；
- [端到端 Benchmark](benchmarks/stream_bench.cu)：CPU Submit、GPU Span、End-to-end、统计量和 CSV；
- [阶段报告](docs/03_stream_report.md)：固定实验合同、正式结果、NSYS 分析顺序与验收答案；
- [V0 原始数据](results/raw/stream_v0_pageable_sync.csv) 与 `results/nsys/stream_v0_pageable_sync*`。

## Profiler 证据边界

当前 AutoDL 容器禁止访问 NVIDIA GPU Performance Counters，NCU 返回 `ERR_NVGPUCTRPERM`，因此不能完成 Bank Conflict、Warp Stall 和 Memory Workload 硬件计数器验收。NSYS 的 CUDA API、Kernel、资源字段、显存活动和时间线已验证并归档。

在未来允许访问计数器的环境中直接运行：

```bash
./scripts/profile_ncu.sh tiled
./scripts/profile_ncu.sh padded
./scripts/profile_reduction_ncu.sh
./scripts/profile_reduction_v3_ncu.sh
```

项目不会用推论或 NSYS 时间线冒充 NCU 硬件指标。
