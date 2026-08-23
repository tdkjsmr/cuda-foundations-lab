#include "cuda_check.cuh"
#include "test_utils.h"
#include "transpose.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuda_foundations::transpose {

// V0 只做连续读取与连续写回，用作后续 Transpose 版本的实际显存带宽参考。
__global__ void copy_kernel(const float* input,
                            float* output,
                            std::size_t width,
                            std::size_t height) {
    // blockIdx.x 选择横向 Tile，threadIdx.x 选择 Tile 内的列。
    const std::size_t x =
        static_cast<std::size_t>(blockIdx.x) * kTileDim + threadIdx.x;

    // blockIdx.y 选择纵向 Tile，threadIdx.y 选择当前线程负责的第一行。
    const std::size_t base_y =
        static_cast<std::size_t>(blockIdx.y) * kTileDim + threadIdx.y;

    // 同一 Warp 的 threadIdx.x 连续，因此相邻线程访问相邻的 FP32 元素。
    for (unsigned int row_offset = 0; row_offset < kTileDim; row_offset += kBlockRows) {
        // 每个线程沿 y 方向间隔 BLOCK_ROWS 处理四行，覆盖完整 32×32 Tile。
        const std::size_t y = base_y + row_offset;

        // 任意矩形及非 32 整除尺寸都可能产生越界线程，读写前必须同时检查 x 和 y。
        if (x < width && y < height) {
            // 行主序索引 y * width + x 对应 input[y][x] 和 output[y][x]。
            const std::size_t linear_index = y * width + x;

            // V0 不做转置，仅执行一次连续 Global Memory 读取和一次连续写入。
            output[linear_index] = input[linear_index];
        }
    }
}

dim3 copy_block_dimensions() {
    // 32 个 x 线程正好覆盖一个 Warp 的连续列，8 个 y 线程组成 256 线程 Block。
    return dim3(kTileDim, kBlockRows, 1U);
}

dim3 copy_grid_dimensions(std::size_t width, std::size_t height) {
    // 零尺寸会让下面的无溢出向上取整公式发生下溢，因此先明确拒绝。
    if (width == 0 || height == 0) {
        throw std::invalid_argument("Copy Kernel 的 Grid 不能对应零尺寸矩阵");
    }

    // 使用 1 + (extent - 1) / tile 向上取整，避免 extent + tile - 1 溢出。
    const std::size_t grid_x = 1U + (width - 1U) / kTileDim;
    const std::size_t grid_y = 1U + (height - 1U) / kTileDim;

    // dim3 分量是 unsigned int；显式拒绝无法表示的 Grid，避免静默截断。
    if (grid_x > std::numeric_limits<unsigned int>::max() ||
        grid_y > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("Copy Kernel 的 Grid 维度超出 dim3 表示范围");
    }

    return dim3(
        static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y), 1U);
}

void launch_copy(const float* device_input,
                 float* device_output,
                 std::size_t width,
                 std::size_t height,
                 cudaStream_t stream) {
    // 空指针和零尺寸属于调用方错误；在 Host 端拒绝无意义的 Kernel Launch。
    if (device_input == nullptr || device_output == nullptr || width == 0 || height == 0) {
        std::cerr << "launch_copy 收到空 Device 指针或零尺寸矩阵" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Block 和 Grid 分开计算，便于测试、Benchmark 与 Nsight 报告记录启动配置。
    const dim3 block = copy_block_dimensions();
    const dim3 grid = copy_grid_dimensions(width, height);

    // 四尖括号语法把 Grid、Block、动态 Shared Memory 大小和 Stream 传给 CUDA Runtime。
    copy_kernel<<<grid, block, 0U, stream>>>(device_input, device_output, width, height);

    // Kernel Launch 异步返回；这里检查启动参数等同步可见错误，但不强制等待 GPU 完成。
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda_foundations::transpose

#ifndef CUDA_FOUNDATIONS_TRANSPOSE_CORE_ONLY

namespace {

// 将 WIDTHxHEIGHT 文本解析为两个非零 size_t，供演示程序快速切换 Shape。
std::pair<std::size_t, std::size_t> parse_shape(const std::string& text) {
    const std::size_t separator = text.find('x');
    if (separator == std::string::npos) {
        throw std::invalid_argument("Shape 必须使用 WIDTHxHEIGHT 格式");
    }

    const std::size_t width = std::stoull(text.substr(0, separator));
    const std::size_t height = std::stoull(text.substr(separator + 1U));
    cuda_foundations::test::checked_element_count(width, height);
    return {width, height};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // 默认使用小矩阵，保证主程序输出易于人工核对；可用 --shape 覆盖。
        std::size_t width = 8;
        std::size_t height = 4;

        // 主程序只接受一个可选的 --shape WIDTHxHEIGHT 参数。
        if (argc == 3 && std::string(argv[1]) == "--shape") {
            const auto shape = parse_shape(argv[2]);
            width = shape.first;
            height = shape.second;
        } else if (argc != 1) {
            std::cerr << "用法: " << argv[0] << " [--shape WIDTHxHEIGHT]" << std::endl;
            return EXIT_FAILURE;
        }

        // 先检查元素数量乘法，再计算字节数，避免 Host 端分配发生整数溢出。
        const std::size_t element_count =
            cuda_foundations::test::checked_element_count(width, height);
        const std::size_t byte_count =
            cuda_foundations::test::checked_float_byte_count(element_count);

        // Host 输入使用确定性数据，CPU Reference 对 V0 执行逐元素复制。
        const std::vector<float> host_input =
            cuda_foundations::test::make_deterministic_input(element_count);
        const std::vector<float> host_reference =
            cuda_foundations::test::copy_reference(host_input);
        std::vector<float> host_output(element_count, 0.0F);

        // 两个裸指针分别保存 Device 输入和输出地址，其生命周期由本函数显式管理。
        float* device_input = nullptr;
        float* device_output = nullptr;

        // 分配连续 Device Buffer；分配和释放不进入 Kernel 性能计时。
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), byte_count));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), byte_count));

        // 将 Host 输入同步复制到 Device 输入缓冲区，方向为 HostToDevice。
        CUDA_CHECK(cudaMemcpy(
            device_input, host_input.data(), byte_count, cudaMemcpyHostToDevice));

        // 启动 V0 Copy Kernel；默认 Stream 中的后续 D2H 会等待此前 Kernel 完成。
        cuda_foundations::transpose::launch_copy(
            device_input, device_output, width, height);

        // 调试/演示路径显式同步，确保异步执行错误在读取结果前被捕获。
        CUDA_CHECK(cudaDeviceSynchronize());

        // 将 Device 输出同步复制回 Host，方向为 DeviceToHost。
        CUDA_CHECK(cudaMemcpy(
            host_output.data(), device_output, byte_count, cudaMemcpyDeviceToHost));

        // V0 是纯数据搬运，包括 NaN 在内都要求位模式逐元素一致。
        std::string error_message;
        const bool correct = cuda_foundations::test::bitwise_equal(
            host_output, host_reference, &error_message);

        // 在退出前按所有权释放两块 Device Buffer。
        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_input));

        if (!correct) {
            std::cerr << "V0 Copy 验证失败: " << error_message << std::endl;
            return EXIT_FAILURE;
        }

        // 只打印少量结果，避免大 Shape 的终端输出掩盖核心信息。
        const std::size_t print_count = std::min<std::size_t>(element_count, 8U);
        std::cout << "V0 Copy 验证通过，Shape=" << width << 'x' << height << "\n前 "
                  << print_count << " 个输出:";
        for (std::size_t index = 0; index < print_count; ++index) {
            std::cout << ' ' << host_output[index];
        }
        std::cout << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        // Host 参数解析或容量检查失败时给出明确原因并返回失败状态。
        std::cerr << "程序失败: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}

#endif
