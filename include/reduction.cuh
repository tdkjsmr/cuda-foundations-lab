#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_foundations::reduction {

// V0 固定使用 256 线程 Block，便于后续版本做单变量对照。
inline constexpr unsigned int kBlockSize = 256U;

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

// 使用 V0 Interleaved Addressing 在 GPU 上反复归约，直到只剩一个 Device 结果。
ReductionLaunchInfo reduce_interleaved(const float* device_input,
                                       float* workspace_a,
                                       float* workspace_b,
                                       std::size_t input_count,
                                       cudaStream_t stream = nullptr);

}  // namespace cuda_foundations::reduction
