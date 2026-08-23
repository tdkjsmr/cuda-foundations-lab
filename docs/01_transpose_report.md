# Transpose 性能报告

## 0. 当前版本与证据边界

- Git 分支：`v1.2`
- V0：Copy Baseline
- V1：Naive Transpose
- V2：Shared Memory Tiled（`tile[32][32]`）
- 下一版本：V3 Padded Shared Memory
- GPU：RTX 3090，`sm_86`
- CUDA Toolkit：12.4
- 正式构建：C++17 / CUDA C++17、Release、`-O3 -lineinfo`

当前 AutoDL Docker 容器禁止访问 NVIDIA GPU Performance Counters。NCU 返回 `ERR_NVGPUCTRPERM`，因此本报告不声称已经测得 DRAM 硬件吞吐率、Global Memory 事务效率、Warp Stall、Occupancy 或 Bank Conflict。

当前可用证据：

```text
CUDA Event：Kernel-only 稳定时间和有效带宽
Compute Sanitizer：越界、非法地址和数据竞争
Nsight Systems：CUDA API、Kernel、H2D/D2H、Stream 与时间线
```

## 1. 可复制命令

### 1.1 Release 构建

```bash
export PATH=/root/.local/bin:$PATH

cmake \
  -S . \
  -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build -j
```

### 1.2 演示与测试

```bash
./build/transpose_v2 --kernel copy --shape 31x33
./build/transpose_v2 --kernel naive --shape 31x33
./build/transpose_v2 --kernel tiled --shape 31x33
ctest --test-dir build --output-on-failure
```

### 1.3 稳定 Benchmark

单个 Shape：

```bash
./build/transpose_bench \
  --kernel all \
  --shape 4096x4096 \
  --warmup 20 \
  --iterations 100 \
  --groups 5
```

正式 Shape 扫描和 CSV：

```bash
./scripts/run_benchmarks.sh
```

### 1.4 Compute Sanitizer

```bash
compute-sanitizer --tool memcheck ./build/transpose_test
compute-sanitizer --tool racecheck ./build/transpose_test
```

### 1.5 Nsight Systems：纯命令行

分析本版本新增的 Tiled Kernel：

```bash
./scripts/profile_nsys.sh tiled
```

分析 Copy Baseline：

```bash
./scripts/profile_nsys.sh copy
```

脚本不启用 `--gpu-metrics-device`，只使用当前容器已验证可用的 CUDA、NVTX 和 OS Runtime 时间线。

## 2. 文件与调用关系

```text
transpose_v2 main
  ├─ 解析 --kernel copy|naive|tiled 与 --shape
  ├─ 创建 Host 输入和对应 CPU Reference
  ├─ cudaMalloc 输入/输出
  ├─ cudaMemcpy HostToDevice
  ├─ launch_copy、launch_naive 或 launch_tiled
  │    ├─ 相同 Grid
  │    ├─ 相同 Block=(32,8)
  │    ├─ 不使用动态 Shared Memory
  │    └─ cudaGetLastError
  ├─ cudaDeviceSynchronize
  ├─ cudaMemcpy DeviceToHost
  ├─ 逐元素位模式比较
  └─ cudaFree

transpose_test
  ├─ 手算验证 CPU Transpose Reference
  ├─ 回归 V0 Copy 与 V1 Naive
  └─ 验证 V2 Tiled 的全部 Shape 与特殊位模式

transpose_bench
  ├─ 同一进程、同一输入、同一 Device Buffer
  ├─ 独立预热 Copy、Naive 和 Tiled
  ├─ CUDA Event 测量各版本
  ├─ 计时后分别验证正确性
  └─ 输出绝对时间、有效带宽和相对指标
```

`src/transpose/transpose.cu` 同时保存三个 Kernel、Host 启动逻辑和演示 `main()`。测试和 Benchmark 编译同一源码的 Core 模式，避免复制 Kernel 实现。

## 3. V0 Copy Baseline

V0 执行：

```cpp
output[y * width + x] = input[y * width + x];
```

同一 Warp 的相邻线程拥有连续 `x`：

```text
input 地址：  ... 0, 1, 2, 3, ...
output 地址： ... 0, 1, 2, 3, ...
```

读写都连续。V0 不做转置，只提供当前 Benchmark 环境下的软件可达 Copy 带宽参考。

## 4. V1 Naive Transpose

V1 执行：

```cpp
input_index  = y * width + x;
output_index = x * height + y;
output[output_index] = input[input_index];
```

### 4.1 为什么读取连续

同一 Warp 中 `y` 相同、`x` 连续：

```text
input[y][0], input[y][1], input[y][2], ...
```

因此输入读取地址连续。

### 4.2 为什么写入跨步

转置输出的行主序 Shape 是 `height × width`。同一 Warp 中 `x` 每增加 1，输出线性地址增加 `height`：

```text
output[0 * height + y]
output[1 * height + y]
output[2 * height + y]
...
```

对于 `4096×4096`，相邻线程写地址相隔：

```text
4096 float × 4 Byte = 16384 Byte
```

V1 的主要实验变量就是这种跨步写。没有引入 Shared Memory、Padding 或其他优化。

### 4.3 3×2 手算

输入 Shape 为 width=3、height=2：

```text
1 2 3
4 5 6
```

行主序输入：

```text
[1, 2, 3, 4, 5, 6]
```

转置后 Shape 为 width=2、height=3：

```text
1 4
2 5
3 6
```

行主序输出：

```text
[1, 4, 2, 5, 3, 6]
```

测试先用这个固定结果验证 CPU Reference，再用 CPU Reference 验证 GPU。

## 5. Grid、Block 与边界

V0/V1 统一使用：

```text
TILE_DIM = 32
BLOCK_ROWS = 8
Block = (32, 8)
Grid = (ceil(width / 32), ceil(height / 32))
```

线程坐标：

```text
x = blockIdx.x × 32 + threadIdx.x
base_y = blockIdx.y × 32 + threadIdx.y
y = base_y + {0, 8, 16, 24}
```

访问前统一检查：

```cpp
if (x < width && y < height)
```

正确性测试覆盖：

```text
1×1
31×33
32×32
33×31
1024×8192
8192×1024
4096×4096
4097×3073
```

特殊值覆盖 `+0`、`-0`、负数、重复值、带 payload 的 NaN、`+Inf` 和 `-Inf`。Transpose 只重排位模式，所以要求逐元素按位一致。

## 6. Benchmark 方法

每个 Kernel 独立执行：

```text
预热 20 次
→ cudaDeviceSynchronize
→ 记录 start Event
→ 连续 Launch 100 次
→ 记录 stop Event
→ 等待 stop Event
→ 总时间除以 100
→ 重复 5 组
→ 计时外 D2H 与 CPU Reference 验证
```

报告最小值、P50、P95 和标准差。计时区间不包含分配、H2D、D2H、CPU Reference 或逐轮同步。

两个版本都读取和写入相同数量的 FP32 元素：

```text
Effective Bandwidth
= 2 × width × height × sizeof(float) / Kernel Time
```

相对指标：

```text
Relative to Copy bandwidth
= 当前版本有效带宽 / Copy 有效带宽 × 100%

Speedup relative to Naive
= Naive P50 / 当前版本 P50
```

## 7. 正式结果

以下结果来自代码提交 `efce964`，不包含 Profiler 开销：

| Kernel | Shape | Min (us) | P50 (us) | P95 (us) | Stddev (us) | GB/s | Copy % | vs Naive |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Copy V0 | 1024×8192 | 82.780 | 82.801 | 82.821 | 0.015 | 810.487 | 100% | 3.141× |
| Naive V1 | 1024×8192 | 260.024 | 260.096 | 260.114 | 0.033 | 258.016 | 31.835% | 1.000× |
| Copy V0 | 8192×1024 | 82.401 | 82.421 | 82.481 | 0.033 | 814.222 | 100% | 2.750× |
| Naive V1 | 8192×1024 | 226.673 | 226.693 | 226.761 | 0.036 | 296.034 | 36.358% | 1.000× |
| Copy V0 | 4096×4096 | 161.106 | 161.126 | 161.180 | 0.031 | 832.996 | 100% | 2.810× |
| Naive V1 | 4096×4096 | 450.222 | 452.833 | 452.940 | 1.040 | 296.395 | 35.582% | 1.000× |
| Copy V0 | 4097×3073 | 121.283 | 121.324 | 121.324 | 0.016 | 830.182 | 100% | 2.934× |
| Naive V1 | 4097×3073 | 355.850 | 355.912 | 355.944 | 0.033 | 282.993 | 34.088% | 1.000× |

原始数据保存在 `results/raw/transpose_v1.csv`。Copy 在四种 Shape 上达到 810–833 GB/s；Naive 只有 Copy 的 31.8%–36.4%，P50 是 Copy 的 2.75–3.14 倍。

相同元素数量下，`1024×8192` 的 Naive 为 260.096 us，而 `8192×1024` 为 226.693 us，说明 Shape 与输出跨度会影响结果。代码地址映射能证明两者都是跨步写，但在 NCU 不可用时，不能进一步把差异归因到某个未经测量的缓存、内存分区或 Warp Stall 指标。

## 8. 已有 V0 NSYS 证据

V0 Smoke Report 成功捕获：

- 6 次 Copy Kernel：5 次预热和 1 次正式 Launch；
- Kernel 平均约 `160.932 us`，中位数约 `161.108 us`；
- H2D 约 `6.586 ms`，D2H 约 `7.016 ms`；
- 稳态 `cudaLaunchKernel` 中位数约 `4.825 us`；
- 首次 `cudaMalloc` 和首次 Launch 存在明显冷启动成本。

这证明 NSYS 的 CUDA API、Kernel 和显存活动采集可用，但不等价于 NCU 的硬件性能计数器分析。

## 9. V1 NSYS 实测与分析顺序

2026-08-23 运行 `./scripts/profile_nsys.sh naive` 得到 `results/nsys/transpose_v1_naive.nsys-rep`。该次配置是 `4096×4096`、Block `(32, 8)`、5 次预热和 1 次正式计时；NSYS 共捕获 6 次 Kernel。

实测数据：

- `cuda_gpu_kern_sum`：6 次 `naive_transpose_kernel` 平均 `510.384 us`，中位数 `510.257 us`，最小 `509.984 us`，最大 `510.912 us`，标准差约 `0.358 us`；
- CUDA Event 记录的唯一正式 Launch 为 `516.416 us`，有效带宽 `259.902 GB/s`；
- `cuda_gpu_mem_time_sum`：H2D `6.201 ms`，D2H `7.233 ms`，两者各搬运 `67.109 MB`；
- `cuda_api_sum`：2 次 `cudaMalloc` 合计约 `161.412 ms`，6 次 `cudaLaunchKernel` 合计约 `166.070 us`，`cudaEventSynchronize` 约 `512.230 us`。

与无 Profiler 的正式 P50 `452.833 us` 相比，NSYS 下的单次 Event 结果高约 14%。因此 `.nsys-rep` 用于验证调用对象、Launch 数量、时序和同步边界，正式性能值仍以无 Profiler 的 5 组 Benchmark 为准。

阅读 NSYS 时依次检查：

1. `cuda_gpu_kern_sum`：确认捕获 6 次 `naive_transpose_kernel`，观察平均、中位数、最小值、最大值和波动；
2. `cuda_gpu_mem_time_sum`：确认 H2D/D2H 大小与 V0 一致，不把传输时间混入 Kernel-only 结论；
3. `cuda_api_sum`：区分首次初始化、稳态 Launch 和同步等待；
4. `cuda_gpu_trace`：确认默认 Stream 上 H2D、6 次 Kernel、D2H 的串行顺序。

`cudaMalloc` 在 API 汇总中占比很高，但它是计时前的一次性初始化；H2D/D2H 也不在 Kernel-only Event 区间。因此这些数字不能用来宣称 Naive Kernel 的主要瓶颈是分配或 PCIe 传输。

V1 当前可以形成的证据链：

```text
代码地址映射显示 V1 连续读、跨步写
→ 同负载 CUDA Event 显示 V1 明显慢于 Copy
→ NSYS 验证被测 Kernel、调用次数和 Device 执行时间
```

不能写成：

```text
NCU 已证明 Global Store 事务效率下降
```

因为当前环境没有取得该硬件指标。

## 10. V2 Shared Memory Tiled Transpose

### 10.1 本版本唯一主要变化

V2 保留 `(32, 8)` Block、相同 Grid、输入、输出和计时规则，只增加 `__shared__ float tile[32][32]`：

```text
连续 Global Load
→ tile[原行][原列]
→ __syncthreads()
→ tile[原列][原行]
→ 连续 Global Store
```

加载阶段，一个 Warp 的 `threadIdx.x` 连续，因此 `input[input_y * width + input_x]` 连续。写回阶段交换 Block 坐标，Warp 写入 `output[output_y * height + output_x]` 的连续 `output_x`。Shared Memory 负责在两种连续 Global Memory 布局之间重排数据。

### 10.2 为什么必须同步

同一线程写入的 Shared Memory 元素可能由另一个线程读取。`__syncthreads()` 同时提供 Block 级执行屏障和 Shared Memory 可见性保证；所有线程都无条件到达屏障，边界判断只包围访问，不包围同步，因此不完整 Tile 也不会发生条件同步死锁。

### 10.3 边界坐标

输入阶段检查：

```text
input_x < width && input_y < height
```

输出逻辑 Shape 是 `width × height`，所以写回阶段检查：

```text
output_x < height && output_y < width
```

这一交换对 `31×33`、`33×31` 和 `4097×3073` 等非整除矩形尤其重要。

### 10.4 本版本刻意保留的 Bank Conflict

V2 的 Shared Memory 行跨度是 32 个 `float`。交换索引后，同一 Warp 读取 `tile[threadIdx.x][固定列]`，相邻线程地址相差 32 个 word，会映射到相同 Bank，形成经典的 Bank Conflict。V2 不加入 Padding；v1.3 将只把布局改为 `tile[32][33]`，验证地址映射变化。

在当前容器不能访问 NCU 计数器的条件下，Bank Conflict 是由地址映射推导出的算法属性，不能写成已经由硬件指标实测证明。

### 10.5 命令

```bash
./build/transpose_v2 --kernel tiled --shape 31x33
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck ./build/transpose_test
compute-sanitizer --tool racecheck ./build/transpose_test
./build/transpose_bench --kernel all --shape 4096x4096 --warmup 20 --iterations 100 --groups 5
./scripts/profile_nsys.sh tiled
```

未来在允许性能计数器的环境中运行：

```bash
./scripts/profile_ncu.sh tiled
```

### 10.6 预览结果与证据边界

正式提交前的 4096×4096 预览显示：Copy P50 `161.516 us`，Naive P50 `449.638 us`，Tiled P50 `168.243 us`。Tiled 达到 Copy 有效带宽的约 96.0%，相对 Naive 加速约 2.67 倍。该结果只用于确认优化方向；正式表格必须在代码提交后重新构建并生成带正确 Git Commit 的 CSV。

## 11. 进入 V3 前需要回答

1. Naive Transpose 是读不连续还是写不连续？
2. Shared Memory 为什么能让 Global Load 和 Store 同时连续？
3. 为什么 `__syncthreads()` 不能只放在边界判断内部？
4. 为什么 `tile[32][32]` 的转置读会发生 Bank Conflict？
5. NSYS 能验证 V2 的哪些事实，又不能验证哪些微架构指标？
