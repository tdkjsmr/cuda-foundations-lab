# Reduction 性能报告

## 0. 当前版本与证据边界

- Git 分支：`v2.1`
- 当前实现：V0 Interleaved + V1 Sequential Addressing
- Block：256 threads
- 数据类型：FP32
- CPU Reference：double 串行累加
- 最终结果：GPU 多阶段归约到一个 Device float
- GPU：RTX 3090，`sm_86`
- 正式构建：C++17 / CUDA C++17、Release、`-O3 -lineinfo`

当前 AutoDL Docker 容器禁止访问 NVIDIA GPU Performance Counters。NCU 返回 `ERR_NVGPUCTRPERM`，因此本报告不声称已经测得 Warp Divergence、Warp Stall、Achieved Occupancy、DRAM Throughput 或 Shared Memory 硬件吞吐。

当前可用证据：

```text
CUDA Event：完整 GPU 多阶段归约的 Kernel-only 时间
CPU double Reference：absolute / normalized error
Compute Sanitizer：越界、数据竞争和同步错误
Nsight Systems：阶段数量、Grid 递减、Kernel 时间和 CUDA API 时间线
```

## 1. 可复制命令

### 1.1 Release 构建

```bash
export PATH=/root/.local/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

### 1.2 演示与测试

```bash
./build/reduction --size 1000003 --pattern random
./build/reduction --size 1025 --pattern dynamic_range
./build/reduction_test
ctest --test-dir build --output-on-failure
```

### 1.3 Compute Sanitizer

```bash
compute-sanitizer --tool memcheck ./build/reduction_test
compute-sanitizer --tool racecheck ./build/reduction_test
compute-sanitizer --tool synccheck ./build/reduction_test
```

### 1.4 稳定 Benchmark

```bash
./build/reduction_bench \
  --kernel interleaved \
  --size 16777219 \
  --warmup 20 \
  --iterations 100 \
  --groups 5

./scripts/run_reduction_benchmarks.sh
```

### 1.5 Profiler

```bash
./scripts/profile_reduction_nsys.sh
```

未来在允许性能计数器的环境运行：

```bash
./scripts/profile_reduction_ncu.sh
```

## 2. 文件与调用关系

```text
reduction main
  ├─ 解析 --size 与 --pattern
  ├─ 生成 Host 输入和 double CPU Reference
  ├─ cudaMalloc 输入与两块 Ping-Pong Workspace
  ├─ cudaMemcpy HostToDevice
  ├─ reduce_interleaved
  │    ├─ Stage 0：N → ceil(N / 256)
  │    ├─ Stage 1：partials → 更少 partials
  │    ├─ ...
  │    └─ Final Stage：partials → 1
  ├─ cudaDeviceSynchronize
  ├─ 只复制一个 float 回 Host
  ├─ absolute / normalized error 验证
  └─ cudaFree

reduction_test
  ├─ 回归全部规定 N
  ├─ zeros / ones / alternating / random / dynamic_range
  ├─ 检查第一阶段 Partial Sum 数量
  ├─ 检查完整 Kernel Launch 数量
  └─ 检查误差阈值

reduction_bench
  ├─ 同一输入和两块 Workspace
  ├─ 每次迭代执行完整多阶段归约
  ├─ Event 只包围全部 Kernel Stage
  ├─ 计时后复制一个结果并验证
  └─ 输出时间、下界带宽、阶段结构和误差
```

`src/reduction/reduction.cu` 同时包含 Kernel、Host 多阶段控制和演示 `main()`；测试与 Benchmark 编译同一源码的 Core 模式，不复制 Kernel 实现。

## 3. V0 Interleaved Addressing

### 3.1 每线程加载一个元素

Block 固定为 256 线程。每个线程读取：

```cpp
global_index = blockIdx.x * blockDim.x + threadIdx.x;
shared[threadIdx.x] = global_index < N ? input[global_index] : 0.0F;
```

最后一个不完整 Block 中，越界线程写入加法单位元 0。这样 Shared Memory 的 256 个位置始终完成初始化，归约阶段不需要再对右操作数做边界判断。

### 3.2 交错活跃线程

V0 使用：

```cpp
for (stride = 1; stride < blockDim.x; stride *= 2) {
    if (threadIdx.x % (2 * stride) == 0) {
        shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
}
```

以一个 Warp 的前几轮为例：

```text
stride=1：lane 0,2,4,6,... 活跃
stride=2：lane 0,4,8,12,... 活跃
stride=4：lane 0,8,16,24 活跃
```

活跃 Lane 分散在 Warp 中，大量 Lane 不执行加法但仍随 Warp 前进。这是 V0 刻意保留的低效寻址基线。

### 3.3 同步位置

每轮加法后都必须执行 `__syncthreads()`，因为下一轮可能由不同线程读取本轮写入。同步位于条件判断外，保证整个 Block 的所有线程无条件到达屏障。

## 4. GPU 多阶段归约

一个 Kernel 只能让每个 Block 产生一个 Partial Sum。Host 控制器继续在两块 Device Workspace 之间 Ping-Pong：

```text
原始输入 → workspace_a → workspace_b → workspace_a → ... → 1 个结果
```

对于 `N=1,000,003`：

```text
Stage 0：1,000,003 → 3,907
Stage 1：3,907 → 16
Stage 2：16 → 1
```

共 3 次 Kernel Launch。Partial Sums 不复制回 CPU，Host 只在最后 D2H 一个 float。

`N=1` 也执行一次 Kernel，使所有输入规模共享同一条 Device 归约路径。

## 5. 数值正确性

CPU 使用 double 串行累加：

```cpp
double reference = 0.0;
for (float value : input) {
    reference += static_cast<double>(value);
}
```

GPU 使用 FP32 树形顺序，不能要求 bitwise 相等。报告：

```text
absolute_error = abs(gpu_result - reference)
normalized_error = absolute_error / max(sum(abs(input)), 1.0)
tolerance = 5e-6 × sum(abs(input)) + 1e-5
```

验收条件：

```text
absolute_error <= tolerance
```

同时保留 absolute 和 normalized error，避免总和接近 0 时只看相对误差失真。

## 6. 测试覆盖

规定 N：

```text
1, 31, 32, 33,
255, 256, 257,
1023, 1024, 1025,
1,000,003, 16,777,219
```

它们覆盖小于/等于/超过 Warp，Block 边界，非 2 的幂，非 Block 整除和大规模多阶段归约。

输入分布：

```text
zeros
ones
alternating
random [-1, 1]
dynamic_range
```

全 1 只用于其中一个用例，不作为唯一正确性依据。

本轮 V0/V1 实际检查结果：

```text
CTest：2/2 PASS（Transpose + Reduction）
memcheck：0 errors
racecheck：0 hazards，0 errors，0 warnings
synccheck：0 errors
```

## 7. Benchmark 语义

正式计时流程：

```text
分配与 H2D
→ 20 次完整归约预热
→ Event Start
→ 连续执行 100 次完整 GPU 多阶段归约
→ Event Stop / Synchronize
→ 总时间除以 100
→ 重复 5 组
→ D2H 一个最终结果并验证
```

计时区间包含所有 Reduction Kernel Stage，不包含 cudaMalloc、H2D、D2H、CPU Reference 和正确性比较。

下界有效带宽：

```text
N × sizeof(float) / 完整归约 P50
```

它不包含中间 Partial Sum 读写，所以明确标记为 Lower-bound Effective Bandwidth。

## 8. 正式结果

测试设备为 RTX 3090。CSV 中 `cuda_driver_api_version=13.2` 是 `cudaDriverGetVersion` 返回的驱动所支持 CUDA Driver API 版本；`cuda_runtime_version=12.4` 是本项目链接的 CUDA Runtime 版本。

| N | P50 (μs) | P95 (μs) | 下界带宽 (GB/s) | 首阶段 Partials | Launches | Absolute Error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.949 | 2.966 | 0.001 | 1 | 1 | 0 |
| 31 | 2.970 | 2.978 | 0.042 | 1 | 1 | 2.570e-7 |
| 32 | 3.369 | 3.406 | 0.038 | 1 | 1 | 1.974e-7 |
| 33 | 2.959 | 2.959 | 0.045 | 1 | 1 | 2.868e-7 |
| 255 | 2.990 | 3.000 | 0.341 | 1 | 1 | 1.753e-8 |
| 256 | 2.980 | 2.996 | 0.344 | 1 | 1 | 6.136e-7 |
| 257 | 5.919 | 5.935 | 0.174 | 2 | 2 | 7.328e-7 |
| 1,023 | 5.868 | 5.876 | 0.697 | 4 | 2 | 1.046e-6 |
| 1,024 | 5.939 | 5.949 | 0.690 | 4 | 2 | 1.463e-6 |
| 1,025 | 5.990 | 5.990 | 0.684 | 5 | 2 | 1.493e-6 |
| 1,000,003 | 28.436 | 28.455 | 140.665 | 3,907 | 3 | 9.518e-5 |
| 16,777,219 | 347.382 | 347.439 | 193.185 | 65,537 | 4 | 1.226e-4 |

全部 12 个规模均满足输入相关 tolerance。`N≤256` 的约 3 μs 主要反映一次 Kernel Launch 的固定成本；`N=257` 增加第二阶段后时间约翻倍。大规模输入才更能反映 V0 Kernel 的吞吐行为。原始数据见 `results/raw/reduction_v0.csv`。

## 9. NSYS 实测分析

命令使用 `N=1,000,003`、5 次预热和 1 次正式归约，共执行 6 次完整归约。报告捕获到 18 次 Kernel，Grid 严格重复：

```text
3907 → 16 → 1
```

按 Grid 汇总：

| Stage Grid | 次数 | 平均 Kernel 时间 | Kernel 总时间占比 |
|---:|---:|---:|---:|
| 3,907 | 6 | 22.430 μs | 81.78% |
| 16 | 6 | 2.539 μs | 9.26% |
| 1 | 6 | 2.460 μs | 8.97% |

其他可验证事实：

- 每个 Kernel 均为 `block=(256,1,1)`、16 registers/thread、1,024 B static shared memory；
- Kernel 总计 164.568 μs；`cudaLaunchKernel` 共 18 次，中位 Host API 时间 3.501 μs；
- H2D 只有一次、约 4 MB，D2H 只有一次、4 B，证明 Partial Sums 没有逐阶段回传 Host；
- H2D 位于归约前，D2H 位于所有 Kernel 后，不在 CUDA Event 的 Kernel-only 正式计时语义内。

NSYS 插桩下 benchmark Event 时间为 34.656 μs，高于无 Profiler 的正式 P50 28.436 μs，因此 NSYS 数字只用于时间线和阶段占比，不替代正式性能基线。NSYS 不能证明 Warp Divergence、Bank Conflict、Stall 原因或实际 DRAM 吞吐；这些仍需未来在允许计数器的环境中用 NCU 验收。

## 10. V1 Sequential Addressing

### 10.1 本轮唯一主要变化

V1 保持以下条件与 V0 完全相同：

```text
每线程加载 1 个元素
Block = 256
Grid = ceil(N / 256)
1,024 B static shared memory
每轮一次 __syncthreads()
GPU 多阶段 Ping-Pong
输入、预热、迭代、计时范围和误差标准
```

唯一主要变化是 Block 内活跃线程的选择方式：

```cpp
for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
        shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
}
```

V0 与 V1 的前几轮对比：

```text
V0 stride=1：lane 0,2,4,... 活跃（同一 Warp 内交错）
V0 stride=2：lane 0,4,8,... 活跃

V1 stride=128：thread 0..127 连续活跃
V1 stride=64： thread 0..63 连续活跃
V1 stride=32： thread 0..31 连续活跃
```

V1 让完全活跃与完全不活跃的 Warp 尽量分离；只有活跃线程数小于一个 Warp 后，最后一个 Warp 才部分活跃。这个源码结构支持“减少 Warp 内分支浪费”的算法假设，但当前环境不能用 NSYS 测出 Warp Divergence，必须等待 NCU 计数器验证。

### 10.2 边界与同步

最后一个不完整 Block 仍由越界线程向 Shared Memory 写 0，因此 `thread_index + stride` 始终落在已初始化的 256 个槽位内。每轮屏障仍在 `if` 外：即使某个线程本轮不做加法，也必须参与 `__syncthreads()`，否则会违反 Block 屏障的一致到达规则。

## 11. V1 公平对比与验证命令

```bash
./build/reduction --kernel interleaved --size 1000003 --pattern random
./build/reduction --kernel sequential --size 1000003 --pattern random
./build/reduction_test

compute-sanitizer --tool memcheck ./build/reduction_test
compute-sanitizer --tool racecheck ./build/reduction_test
compute-sanitizer --tool synccheck ./build/reduction_test

./scripts/run_reduction_v1_benchmarks.sh
./scripts/profile_reduction_v1_nsys.sh
```

正式 CSV 对每个规定 N 依次运行 Interleaved 与 Sequential；两版本使用同一个二进制、同一 deterministic random 输入和相同的 `20 × 100 × 5` 统计方法。

## 12. V1 正式结果与 NSYS 证据

### 12.1 CUDA Event 正式对比

测试设备、输入、Block、Workspace、多阶段调度和 `20 × 100 × 5` 统计方法完全相同。P50 结果如下：

| N | V0 Interleaved (μs) | V1 Sequential (μs) | V1 Speedup | V1 P95 (μs) |
|---:|---:|---:|---:|---:|
| 1 | 2.970 | 2.437 | 1.219× | 2.456 |
| 31 | 2.990 | 2.406 | 1.243× | 2.431 |
| 32 | 2.970 | 2.406 | 1.234× | 2.456 |
| 33 | 2.990 | 2.396 | 1.248× | 2.406 |
| 255 | 2.877 | 2.396 | 1.201× | 2.396 |
| 256 | 2.877 | 2.406 | 1.196× | 2.492 |
| 257 | 5.714 | 5.069 | 1.127× | 5.515 |
| 1,023 | 5.837 | 4.854 | 1.203× | 4.887 |
| 1,024 | 5.888 | 5.274 | 1.117× | 19.628 |
| 1,025 | 5.919 | 4.792 | 1.235× | 4.833 |
| 1,000,003 | 28.140 | 20.347 | 1.383× | 20.355 |
| 16,777,219 | 347.310 | 211.343 | 1.643× | 211.403 |

V1 在全部规定规模的 P50 上均更快。大输入的收益更明显：`N=1,000,003` 降低 27.69%，`N=16,777,219` 降低 39.15%。小输入仍主要由 1～2 次 Launch 的固定成本主导，因此绝对收益有限。

`N=1,024` 的 V1 P95 为 19.628 μs，明显偏离其 5.274 μs P50 及相邻规模；原始波动被如实保留，没有筛除。它不改变 P50 的优化方向，但说明小任务容易被系统抖动放大。原始 24 条记录见 `results/raw/reduction_v1_comparison.csv`。

全部结果满足题目规定的输入相关 tolerance。V1 的 `dynamic_range` 用例由于 `±1e8` 强消去和不同 FP32 加法顺序，absolute error 为 32.25，但 normalized error 仅 `9.430e-10`；随机、全零、全一、正负交替及全部边界 N 同时通过。

### 12.2 NSYS 阶段对比

V1 报告捕获到 18 次 `sequential_reduction_kernel`，严格重复 6 组：

```text
3907 → 16 → 1
```

| Grid | V0 平均 Kernel (μs) | V1 平均 Kernel (μs) | 阶段 Speedup | V1 时间占比 |
|---:|---:|---:|---:|---:|
| 3,907 | 22.430 | 15.664 | 1.432× | 80.97% |
| 16 | 2.539 | 1.881 | 1.350× | 9.72% |
| 1 | 2.460 | 1.801 | 1.366× | 9.31% |

18 次 Kernel 总时间由 V0 的 164.568 μs 降至 V1 的 116.075 μs，NSYS 阶段汇总加速 `1.418×`。两版本仍为 16 registers/thread、1,024 B static shared memory、同样的 Grid/Block 和 18 次 Launch，说明测得的下降不是由减少阶段、Block 或 Shared Memory 容量造成。

V1 仍只有一次约 4 MB H2D 和一次 4 B D2H，中间 Partial Sums 没有回到 CPU。`cudaLaunchKernel` 中位 Host API 时间为 3.413 μs。NSYS 插桩下 Event 时间为 26.368 μs，不能替代无 Profiler 的正式 P50。

结论边界：源码与性能数据符合“Sequential 减少 Warp 内无效分支工作”的假设，但 NSYS 不提供 Branch Efficiency、Warp Stall、Achieved Occupancy 或 DRAM Throughput。当前 Docker 无 NCU 计数器权限，因此不把这一因果解释写成已由硬件计数器证明的事实。

## 13. 进入 V2 前需要回答

1. V0 和 V1 每轮分别有哪些线程活跃？
2. 为什么 V1 的活跃线程更适合 Warp 的 SIMT 执行？
3. V1 是否减少了同步次数、Launch 次数或 Shared Memory 容量？
4. 为什么 `thread_index + stride` 对 256 线程 Block 始终合法？
5. NSYS 能验证 V1 的哪些结构事实，不能验证哪些 Warp 行为？
6. 下一轮每线程加载两个元素为什么会减少首阶段 Block 数？
