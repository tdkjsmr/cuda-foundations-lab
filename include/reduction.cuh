#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_foundations::reduction {

// V0/V1 固定使用 256 线程 Block，保证版本对比只改变归约寻址方式。
inline constexpr unsigned int kBlockSize = 256U;

// 标识同一多阶段框架中实际使用的 Block 内归约算法。
enum class KernelVersion {
    kInterleaved,
    kSequential,
};

// 记录一次完整 GPU 多阶段归约的结果位置与结构信息。
struct ReductionLaunchInfo {
    // device_result 指向 workspace_a 或 workspace_b 中的唯一最终结果。
    const float* device_result = nullptr;

    // first_stage_partial_count 是第一阶段产生的 Partial Sum 数量。
    std::size_t first_stage_partial_count = 0U;

    // launch_count 是从原始输入归约到单一结果的 Kernel Launch 总数。
    std::size_t launch_count = 0U;
};

// 返回容纳第一阶段 Partial Sums 所需的 workspace 元素数。
std::size_t workspace_elements(std::size_t input_count);

// 返回 V0 处理给定 N 所需的完整 GPU Kernel Launch 数。
std::size_t interleaved_launch_count(std::size_t input_count);

// 返回 V1 处理给定 N 所需的完整 GPU Kernel Launch 数。
std::size_t sequential_launch_count(std::size_t input_count);

// 返回适合终端和 CSV 的稳定 Kernel 名称。
const char* kernel_name(KernelVersion version);

// 使用 V0 Interleaved Addressing 在 GPU 上反复归约，直到只剩一个 Device 结果。
ReductionLaunchInfo reduce_interleaved(const float* device_input,
                                       float* workspace_a,
                                       float* workspace_b,
                                       std::size_t input_count,
                                       cudaStream_t stream = nullptr);

// 使用 V1 Sequential Addressing 在 GPU 上反复归约，直到只剩一个结果。
ReductionLaunchInfo reduce_sequential(const float* device_input,
                                      float* workspace_a,
                                      float* workspace_b,
                                      std::size_t input_count,
                                      cudaStream_t stream = nullptr);

// 按 version 调度 V0 或 V1；测试和 Benchmark 用它保证两版本路径一致。
ReductionLaunchInfo reduce(KernelVersion version,
                           const float* device_input,
                           float* workspace_a,
                           float* workspace_b,
                           std::size_t input_count,
                           cudaStream_t stream = nullptr);

}  // namespace cuda_foundations::reduction
