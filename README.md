# CUDA Foundations Lab

这是 OLCF CUDA Training Series 阶段收尾项目，目标是建立“正确性、稳定测量、Profiler 证据、单变量迭代”的 CUDA 性能工程闭环。

当前开发分支：`v2.2`。

Transpose 子项目最终分支：`v1.3`。

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
src/reduction/reduction.cu  V0/V1/V2 Kernel、GPU 多阶段控制和演示 main
tests/transpose_test.cu     CPU Reference、正确性、边界和特殊位模式测试
tests/reduction_test.cu     规定 N、数据分布、误差与多阶段测试
benchmarks/                 独立 Copy/Naive/Tiled/Padded 稳定态性能测量
                            以及 Reduction 完整多阶段稳定 Benchmark
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

## Reduction V0/V1/V2 当前状态

- Interleaved Addressing：已保留为 V0 基线；
- Sequential Addressing：已实现为 V1，连续前半线程参与归约；
- First Add During Load：已实现为 V2，每线程最多先合并两个输入；
- 任意 N 和非 2 的幂：已支持；
- GPU 多阶段归约：已支持，两块 Workspace Ping-Pong 到单个结果；
- CPU Reference：double 串行累加；
- 误差：absolute error、normalized error 和输入相关 tolerance；
- 测试：覆盖全部规定 N 与五种数据分布；
- Benchmark：V1 在 N=1,000,003 和 16,777,219 上分别较 V0 加速 1.383× 和 1.643×；
- NSYS：V1 三阶段结构已验证，报告与命令行 CSV 已归档；NCU 仍受容器权限限制。

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

./build/reduction --size 1000003 --pattern random
./scripts/run_reduction_v2_benchmarks.sh
./scripts/profile_reduction_v2_nsys.sh
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

## 最终交付物

- [统一源码](src/transpose/transpose.cu)：四个 Kernel、Host Launcher 与演示 `main()`；
- [正确性测试](tests/transpose_test.cu)：CPU Reference、全部 Shape 和特殊位模式；
- [稳定 Benchmark](benchmarks/transpose_bench.cu)：CUDA Event、统计量和 CSV；
- [完整性能报告](docs/01_transpose_report.md)：实现原理、结果、NSYS 证据与口头验收答案；
- [最终原始数据](results/raw/transpose_v3.csv)：四版本、四种正式 Shape；
- `results/nsys/`：各阶段 `.nsys-rep` 与命令行 CSV 摘要。

Reduction V0/V1/V2：

- [统一源码](src/reduction/reduction.cu)：Interleaved/Sequential/First Add Kernel、GPU 多阶段 Host 控制与演示 `main()`；
- [正确性测试](tests/reduction_test.cu)：全部规定 N、五类输入和误差验收；
- [稳定 Benchmark](benchmarks/reduction_bench.cu)：完整多阶段 CUDA Event 计时；
- [性能报告](docs/02_reduction_report.md)：算法、测试、正式数据与 NSYS 阶段分析；
- [V0 原始数据](results/raw/reduction_v0.csv) 与 `results/nsys/reduction_v0_interleaved*`。
- [V1 对比数据](results/raw/reduction_v1_comparison.csv) 与 `results/nsys/reduction_v1_sequential*`。

## Profiler 证据边界

当前 AutoDL 容器禁止访问 NVIDIA GPU Performance Counters，NCU 返回 `ERR_NVGPUCTRPERM`，因此不能完成 Bank Conflict、Warp Stall 和 Memory Workload 硬件计数器验收。NSYS 的 CUDA API、Kernel、资源字段、显存活动和时间线已验证并归档。

在未来允许访问计数器的环境中直接运行：

```bash
./scripts/profile_ncu.sh tiled
./scripts/profile_ncu.sh padded
./scripts/profile_reduction_v2_ncu.sh
```

项目不会用推论或 NSYS 时间线冒充 NCU 硬件指标。
