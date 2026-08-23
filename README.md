# CUDA Foundations Lab

这是 OLCF CUDA Training Series 阶段收尾项目，目标是建立“正确性、稳定测量、Profiler 证据、单变量迭代”的 CUDA 性能工程闭环。

当前分支：`v1.3`。

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
tests/transpose_test.cu     CPU Reference、正确性、边界和特殊位模式测试
benchmarks/                 独立 Copy/Naive/Tiled/Padded 稳定态性能测量
scripts/                    测试、Benchmark、NCU/NSYS 命令
results/                    原始 CSV 与 Profiler 报告
docs/                       中文实现与性能分析报告
```

## 配置、构建和运行

```bash
export PATH=/root/.local/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j

./build/transpose_v3 --kernel padded --shape 31x33
ctest --test-dir build --output-on-failure
./scripts/run_benchmarks.sh
./scripts/profile_nsys.sh padded
```

当前 AutoDL 容器禁止访问 NVIDIA GPU Performance Counters，因此 NCU 不能用于微架构指标验收。NSYS 的 CUDA API、Kernel、显存活动和时间线分析已验证可用；详细边界与命令见 `docs/01_transpose_report.md`。
