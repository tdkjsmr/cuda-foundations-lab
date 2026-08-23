# CUDA Foundations Lab

这是 OLCF CUDA Training Series 阶段收尾项目，目标是建立“正确性、稳定测量、Profiler 证据、单变量迭代”的 CUDA 性能工程闭环。

当前分支：`v1.0`。

当前只实现 Transpose V0 Copy Baseline：

```text
input[y][x] → output[y][x]
```

V0 不执行转置。它用于测量当前 GPU、编译参数和 Benchmark 方法下连续 Global Memory 读写的实际参考带宽。Naive Transpose、Shared Memory Tiled 和 Padding 分别在后续 `v1.1`、`v1.2`、`v1.3` 增加。

## 目录关系

```text
include/                    CUDA 错误检查、Event 计时、统计和公共接口
src/transpose/transpose.cu  V0 Kernel、Host 启动逻辑和演示 main
tests/transpose_test.cu     正确性、边界和特殊位模式测试
benchmarks/                 独立稳定态性能测量
scripts/                    构建后测试、Benchmark 和 NCU 命令
results/                    原始 CSV 与 Nsight 报告
docs/                       中文实现与性能分析报告
```

## 配置、构建和运行

```bash
export PATH=/root/.local/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
./build/transpose_v0 --shape 31x33
ctest --test-dir build --output-on-failure
./scripts/run_benchmarks.sh
```

详细的数据流、逐段代码解释、Benchmark 方法与 Nsight Compute 命令见 `docs/01_transpose_report.md`。
