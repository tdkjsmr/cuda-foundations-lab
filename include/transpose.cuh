#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_foundations::transpose {

// TILE_DIM 与后续 Shared Memory Transpose 保持一致，便于公平比较各版本。
inline constexpr unsigned int kTileDim = 32;

// 一个 Block 在 y 方向只启用 8 行线程，每个线程循环处理 4 行数据。
inline constexpr unsigned int kBlockRows = 8;

// 返回 Transpose 各版本统一使用的二维线程块配置：(32, 8)，共 256 个线程。
dim3 transpose_block_dimensions();

// 根据任意矩形的宽高计算覆盖全部输入元素的二维 Grid。
dim3 transpose_grid_dimensions(std::size_t width, std::size_t height);

// 启动 V0 Copy Kernel；输入输出均为 width × height 的连续 FP32 Device Buffer。
void launch_copy(const float* device_input,
                 float* device_output,
                 std::size_t width,
                 std::size_t height,
                 cudaStream_t stream = nullptr);

// 启动 V1 Naive Transpose；输入为 width × height，输出为 height × width。
void launch_naive(const float* device_input,
                  float* device_output,
                  std::size_t width,
                  std::size_t height,
                  cudaStream_t stream = nullptr);

}  // namespace cuda_foundations::transpose
