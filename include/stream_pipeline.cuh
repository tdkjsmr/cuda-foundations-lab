#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

namespace cuda_foundations::streams {

// Stream 实验始终复用 Transpose 最终版的 32 x 8 Thread Block。
inline constexpr unsigned int kTileDim = 32U;
inline constexpr unsigned int kBlockRows = 8U;
// Padding 使转置读取 Shared Memory 时不再是 32-way Bank Conflict。
inline constexpr unsigned int kPaddedTileColumns = kTileDim + 1U;

// 一次 Pipeline Iteration 处理 chunk_count 张相互独立的矩阵。
struct StreamWorkload {
    std::size_t width = 4096U;
    std::size_t height = 2048U;
    std::size_t chunk_count = 16U;
};

// 校验后的派生布局集中保存容量；执行入口仍会核对它与 Workload 是否一致。
struct WorkloadLayout {
    std::size_t elements_per_chunk = 0U;
    std::size_t bytes_per_chunk = 0U;
    std::size_t total_elements = 0U;
    std::size_t total_bytes = 0U;
};

// V3.0/V3.1 同步基线复用一对 Chunk-sized Device Buffer。
struct DeviceBufferPair {
    float* input = nullptr;
    float* output = nullptr;
    std::size_t capacity_bytes = 0U;

    DeviceBufferPair() = default;
    ~DeviceBufferPair();
    DeviceBufferPair(const DeviceBufferPair&) = delete;
    DeviceBufferPair& operator=(const DeviceBufferPair&) = delete;
    DeviceBufferPair(DeviceBufferPair&& other) noexcept;
    DeviceBufferPair& operator=(DeviceBufferPair&& other) noexcept;
};

// V3.1 用一对整批大小的 Pinned Host Buffer 替换 std::vector Pageable Buffer。
struct PinnedHostBufferPair {
    float* input = nullptr;
    float* output = nullptr;
    std::size_t capacity_bytes = 0U;

    // Pinned Memory 是独占资源，因此禁止复制，只允许安全转移所有权。
    PinnedHostBufferPair() = default;
    ~PinnedHostBufferPair();
    PinnedHostBufferPair(const PinnedHostBufferPair&) = delete;
    PinnedHostBufferPair& operator=(const PinnedHostBufferPair&) = delete;
    PinnedHostBufferPair(PinnedHostBufferPair&& other) noexcept;
    PinnedHostBufferPair& operator=(PinnedHostBufferPair&& other) noexcept;
};

// 这些字段只说明硬件具备潜在并发能力，不是已经发生重叠的证据。
struct DeviceCapabilities {
    int device_id = 0;
    std::string device_name;
    int device_overlap = 0;
    int async_engine_count = 0;
    int concurrent_kernels = 0;
};

// 检查 Shape、Chunk 数和所有 size_t 乘法，并返回完整布局。
WorkloadLayout validate_and_derive_layout(const StreamWorkload& workload);

// 读取当前 CUDA Device 的 Stream/拷贝并发能力字段。
DeviceCapabilities query_device_capabilities();

// 为一个 Chunk 分配独立的 Device Input 和 Device Output。
DeviceBufferPair allocate_device_buffers(std::size_t capacity_bytes);

// 在正常路径中逆序释放 Device Buffer，并将指针清空。
void release_device_buffers(DeviceBufferPair* buffers);

// 使用 cudaMallocHost 为整批 Input/Output 分配 Page-locked Host Memory。
PinnedHostBufferPair allocate_pinned_host_buffers(std::size_t capacity_bytes);

// 使用 cudaFreeHost 逆序释放两块 Pinned Buffer，并清空所有权状态。
void release_pinned_host_buffers(PinnedHostBufferPair* buffers);

// 启动从 Transpose V3 冻结下来的 Padded Tiled Kernel。
void launch_padded_tiled_transpose(const float* input,
                                  float* output,
                                  std::size_t width,
                                  std::size_t height,
                                  cudaStream_t stream = nullptr);

// 对 Pageable 或 Pinned Host 指针执行相同的 blocking H2D -> Kernel -> D2H。
void execute_synchronous_pipeline(const StreamWorkload& workload,
                                  const WorkloadLayout& layout,
                                  const float* host_input,
                                  float* host_output,
                                  const DeviceBufferPair& buffers,
                                  bool annotate_nvtx = false);

}  // namespace cuda_foundations::streams
