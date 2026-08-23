# Transpose 性能报告

## 0. 当前版本

- Git 分支：`v1.0`
- 实现：V0 Copy Baseline
- 目标：建立连续读取、连续写入的实际参考带宽
- 尚未实现：Naive Transpose、Shared Memory Tile、Padding

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

正式构建使用 CUDA C++17、`-O3`、`-lineinfo` 和 `sm_86`，不启用 `-G` 或 `--use_fast_math`。

### 1.2 演示与测试

```bash
./build/transpose_v0 --shape 31x33
ctest --test-dir build --output-on-failure
```

### 1.3 稳定 Benchmark

```bash
./build/transpose_bench \
  --kernel copy \
  --shape 4096x4096 \
  --warmup 20 \
  --iterations 100 \
  --groups 5 \
  --csv results/raw/transpose_v0.csv
```

### 1.4 Compute Sanitizer

```bash
compute-sanitizer --tool memcheck ./build/transpose_test
compute-sanitizer --tool racecheck ./build/transpose_test
```

### 1.5 Nsight Compute：纯命令行采集

```bash
ncu \
  --launch-skip 20 \
  --launch-count 1 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section LaunchStats \
  --section Occupancy \
  --section WarpStateStats \
  --import-source yes \
  --export results/ncu/transpose_v0_copy \
  --force-overwrite \
  ./build/transpose_bench \
  --kernel copy \
  --shape 4096x4096 \
  --warmup 20 \
  --profile
```

终端查看和 CSV 导出：

```bash
ncu --import results/ncu/transpose_v0_copy.ncu-rep --page details

ncu \
  --import results/ncu/transpose_v0_copy.ncu-rep \
  --page raw \
  --csv \
  --log-file results/ncu/transpose_v0_copy_raw.csv
```

## 2. 文件与调用关系

```text
transpose_v0 main
  ├─ 创建 Host 输入与 CPU Copy Reference
  ├─ cudaMalloc 两个 Device Buffer
  ├─ cudaMemcpy HostToDevice
  ├─ launch_copy
  │    ├─ 计算 Grid=(ceil(width/32), ceil(height/32))
  │    ├─ Block=(32, 8)
  │    ├─ copy_kernel<<<grid, block>>>
  │    └─ cudaGetLastError
  ├─ cudaDeviceSynchronize
  ├─ cudaMemcpy DeviceToHost
  ├─ 位模式比较
  └─ cudaFree

transpose_test
  └─ 调用同一个 launch_copy，覆盖全部指定 Shape 和特殊位模式

transpose_bench
  └─ 调用同一个 launch_copy，用 CUDA Event 测量多组稳定态 Kernel 时间
```

`src/transpose/transpose.cu` 同时保存 Kernel、Host 启动逻辑和演示 `main()`。测试和 Benchmark 复用其中的 `launch_copy`，但分别拥有自己的入口，因此测试逻辑不会污染正式计时路径。

## 3. V0 Kernel 数据映射

线程块为 `(32, 8)`，共 256 个线程。一个 Block 覆盖一个 `32×32` Tile：

```text
x = blockIdx.x × 32 + threadIdx.x
base_y = blockIdx.y × 32 + threadIdx.y
y = base_y + {0, 8, 16, 24}
```

每个线程处理同一列的四个元素。一个 Warp 内 `threadIdx.x` 从 0 到 31，因此每次循环中相邻线程读取和写入相邻 FP32 地址，形成合并访问。

核心操作是：

```cpp
output[y * width + x] = input[y * width + x];
```

V0 没有转置，也没有 Shared Memory。它只回答一个问题：在完全连续的访问模式下，当前 Benchmark 环境实际能达到多少有效显存带宽。

## 4. 边界处理

Grid 对宽高分别向上取整。当 Shape 不是 32 的整数倍时，最后一个 Tile 会包含无效线程，因此每次 Global Memory 访问前都检查：

```cpp
if (x < width && y < height)
```

测试覆盖：

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

另外使用 `+0`、`-0`、负数、重复值、带 payload 的 NaN、`+Inf` 和 `-Inf`。因为 Copy 不改变数值，测试通过 IEEE-754 位模式比较，而不是使用 `NaN == NaN`。

## 5. Benchmark 方法

正式计时前完成 Context 初始化、Device 分配、输入生成和 H2D。流程为：

```text
预热 20 次
→ 记录 start Event
→ 连续 Launch 100 次
→ 记录 stop Event
→ 等待 stop Event
→ 总时间除以 100
→ 重复 5 组
```

报告 5 组单次平均耗时的最小值、P50、P95 和标准差。计时区间内不执行 `cudaDeviceSynchronize()`，避免人为插入逐轮同步开销。

FP32 Copy 每个元素读取 4 Byte、写入 4 Byte：

```text
Effective Bandwidth = 2 × width × height × sizeof(float) / Kernel Time
```

这里使用十进制 GB/s。它是软件可达的 Copy 参考带宽，不等同于显卡规格表中的理论峰值。

## 6. 实测结果

运行正式 Benchmark 后填写，禁止预先编造：

| Shape | Min (us) | P50 (us) | P95 (us) | Stddev (us) | Effective GB/s |
|---|---:|---:|---:|---:|---:|
| 4096×4096 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 7. V0 的 Nsight 验收问题

收到 NCU 报告后至少回答：

1. DRAM Throughput 达到峰值的多少比例？
2. Global Load/Store 是否形成合并访问？
3. 主要 Warp Stall 是什么，是否真的阻止调度器持续发射？
4. `(32,8)` Block 的 Registers、Shared Memory 和理论/实际 Occupancy 是多少？
5. 长条矩阵、宽条矩阵和方阵的 Copy 带宽是否存在稳定差异？

这些答案必须来自实际 NCU 指标，不在采集前预设结论。
