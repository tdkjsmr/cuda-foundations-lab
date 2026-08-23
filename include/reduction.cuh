#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_foundations::reduction {

// V0/V1/V2/V3 固定使用 256 线程 Block，保持四个版本使用同一个线程块大小。
inline constexpr unsigned int kBlockSize = 256U;

// 标识同一多阶段框架中实际使用的 Block 内归约算法。
enum class KernelVersion {
    kInterleaved,
    kSequential,
    kFirstAdd,
    kWarpShuffle,
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

// 返回 V0/V1 容纳第一阶段 Partial Sums 所需的 Workspace 元素数。
std::size_t workspace_elements(std::size_t input_count);

// 按版本返回 Workspace 元素数；V2/V3 每个 Block 覆盖 512 个输入。
std::size_t workspace_elements(KernelVersion version, std::size_t input_count);

// 返回 V0 处理给定 N 所需的完整 GPU Kernel Launch 数。
std::size_t interleaved_launch_count(std::size_t input_count);

// 返回 V1 处理给定 N 所需的完整 GPU Kernel Launch 数。
std::size_t sequential_launch_count(std::size_t input_count);

// 返回 V2 处理给定 N 所需的完整 GPU Kernel Launch 数。
std::size_t first_add_launch_count(std::size_t input_count);

// 返回 V3 处理给定 N 所需的完整 GPU Kernel Launch 数。
std::size_t warp_shuffle_launch_count(std::size_t input_count);

// 按版本返回从输入归约到一个结果所需的完整 Launch 数。
std::size_t reduction_launch_count(KernelVersion version, std::size_t input_count);

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

// 使用 V2 First Add During Load 完成完整 GPU 多阶段归约。
ReductionLaunchInfo reduce_first_add(const float* device_input,
                                     float* workspace_a,
                                     float* workspace_b,
                                     std::size_t input_count,
                                     cudaStream_t stream = nullptr);

// 使用 V3 Warp Shuffle 完成完整 GPU 多阶段归约。
ReductionLaunchInfo reduce_warp_shuffle(const float* device_input,
                                        float* workspace_a,
                                        float* workspace_b,
                                        std::size_t input_count,
                                        cudaStream_t stream = nullptr);

// 按 version 调度 V0～V3；测试与 Benchmark 共用同一控制路径。
ReductionLaunchInfo reduce(KernelVersion version,
                           const float* device_input,
                           float* workspace_a,
                           float* workspace_b,
                           std::size_t input_count,
                           cudaStream_t stream = nullptr);

}  // namespace cuda_foundations::reduction
