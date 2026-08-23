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

        // 任意矩形及非 32 整除尺寸都可能产生越界线程。
        if (x < width && y < height) {
            // 行主序索引 y * width + x 对应 input[y][x] 和 output[y][x]。
            const std::size_t linear_index = y * width + x;

            // V0 不做转置，仅执行一次连续 Global Memory 读取和一次连续写入。
            output[linear_index] = input[linear_index];
        }
    }
}

// V1 直接把 input[y][x] 写到 output[x][y]，不使用 Shared Memory。
__global__ void naive_transpose_kernel(const float* input,
                                       float* output,
                                       std::size_t width,
                                       std::size_t height) {
    // x 由 threadIdx.x 展开；同一 Warp 的相邻线程拥有连续 x。
    const std::size_t x =
        static_cast<std::size_t>(blockIdx.x) * kTileDim + threadIdx.x;

    // base_y 是当前线程在输入 Tile 内负责的第一行。
    const std::size_t base_y =
        static_cast<std::size_t>(blockIdx.y) * kTileDim + threadIdx.y;

    // 一个线程以 8 行为步长处理四个输入元素，保持与 V0 相同的 Block 配置。
    for (unsigned int row_offset = 0; row_offset < kTileDim; row_offset += kBlockRows) {
        // y 表示输入矩阵的行号。
        const std::size_t y = base_y + row_offset;

        // 最后一个不完整 Tile 中，只有落在原矩阵范围内的线程才能访问显存。
        if (x < width && y < height) {
            // 同一 Warp 的 x 连续，所以 input[y][x] 的读取地址连续。
            const std::size_t input_index = y * width + x;

            // 转置后输出 Shape 为 height × width，output[x][y] 的行跨度是 height。
            const std::size_t output_index = x * height + y;

            // 同一 Warp 的不同 x 导致写地址相隔 height 个 float，形成跨步写入。
            output[output_index] = input[input_index];
        }
    }
}

dim3 transpose_block_dimensions() {
    // 32 个 x 线程正好覆盖一个 Warp 的连续列，8 个 y 线程组成 256 线程 Block。
    return dim3(kTileDim, kBlockRows, 1U);
}

dim3 transpose_grid_dimensions(std::size_t width, std::size_t height) {
    // 零尺寸会让下面的无溢出向上取整公式发生下溢，因此先明确拒绝。
    if (width == 0 || height == 0) {
        throw std::invalid_argument("Transpose Kernel 的 Grid 不能对应零尺寸矩阵");
    }

    // 使用 1 + (extent - 1) / tile 向上取整，避免 extent + tile - 1 溢出。
    const std::size_t grid_x = 1U + (width - 1U) / kTileDim;
    const std::size_t grid_y = 1U + (height - 1U) / kTileDim;

    // dim3 分量是 unsigned int；显式拒绝无法表示的 Grid，避免静默截断。
    if (grid_x > std::numeric_limits<unsigned int>::max() ||
        grid_y > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("Transpose Kernel 的 Grid 维度超出 dim3 表示范围");
    }

    return dim3(
        static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y), 1U);
}

// 在 Host 端统一拒绝空 Device 指针和零尺寸矩阵。
void validate_launch_arguments(const float* device_input,
                               const float* device_output,
                               std::size_t width,
                               std::size_t height,
                               const char* launcher_name) {
    if (device_input == nullptr || device_output == nullptr || width == 0 || height == 0) {
        std::cerr << launcher_name << " 收到空 Device 指针或零尺寸矩阵" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void launch_copy(const float* device_input,
                 float* device_output,
                 std::size_t width,
                 std::size_t height,
                 cudaStream_t stream) {
    // V0 和 V1 共享相同的参数检查、Block 形状与输入 Grid，保证比较公平。
    validate_launch_arguments(device_input, device_output, width, height, "launch_copy");
    const dim3 block = transpose_block_dimensions();
    const dim3 grid = transpose_grid_dimensions(width, height);

    // 动态 Shared Memory 为 0；stream 指定任务进入哪个 CUDA Stream。
    copy_kernel<<<grid, block, 0U, stream>>>(device_input, device_output, width, height);

    // 只检查异步 Launch 的立即错误，不在 Benchmark 的每次 Launch 后强制同步。
    CUDA_CHECK(cudaGetLastError());
}

void launch_naive(const float* device_input,
                  float* device_output,
                  std::size_t width,
                  std::size_t height,
                  cudaStream_t stream) {
    // V1 使用与 V0 完全相同的 Grid/Block，仅改变输出索引这一项主要因素。
    validate_launch_arguments(device_input, device_output, width, height, "launch_naive");
    const dim3 block = transpose_block_dimensions();
    const dim3 grid = transpose_grid_dimensions(width, height);

    // V1 不使用 Shared Memory，直接从连续输入位置写到转置后的跨步输出位置。
    naive_transpose_kernel<<<grid, block, 0U, stream>>>(
        device_input, device_output, width, height);

    // 捕获无效配置等 Launch 错误；执行阶段错误由调用方的同步点捕获。
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda_foundations::transpose

#ifndef CUDA_FOUNDATIONS_TRANSPOSE_CORE_ONLY

namespace {

// 演示程序默认运行 V1，也允许显式回看 V0。
struct DemoOptions {
    std::string kernel_name = "naive";
    std::size_t width = 8U;
    std::size_t height = 4U;
};

// 将 WIDTHxHEIGHT 文本解析为两个非零 size_t。
std::pair<std::size_t, std::size_t> parse_shape(const std::string& text) {
    const std::size_t separator = text.find('x');
    if (separator == std::string::npos || separator == 0U || separator + 1U >= text.size()) {
        throw std::invalid_argument("Shape 必须使用 WIDTHxHEIGHT 格式");
    }

    std::size_t width_characters = 0U;
    std::size_t height_characters = 0U;
    const std::string width_text = text.substr(0U, separator);
    const std::string height_text = text.substr(separator + 1U);
    const std::size_t width = std::stoull(width_text, &width_characters);
    const std::size_t height = std::stoull(height_text, &height_characters);

    // stoull 允许尾随字符，因此必须确认两个子串都被完整消费。
    if (width_characters != width_text.size() || height_characters != height_text.size()) {
        throw std::invalid_argument("Shape 包含无法解析的尾随字符");
    }

    cuda_foundations::test::checked_element_count(width, height);
    return {width, height};
}

// 解析 --kernel copy|naive 和 --shape WIDTHxHEIGHT，参数顺序可以互换。
DemoOptions parse_options(int argc, char** argv) {
    DemoOptions options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "--kernel" && index + 1 < argc) {
            options.kernel_name = argv[++index];
            if (options.kernel_name != "copy" && options.kernel_name != "naive") {
                throw std::invalid_argument("--kernel 只支持 copy 或 naive");
            }
        } else if (argument == "--shape" && index + 1 < argc) {
            const auto shape = parse_shape(argv[++index]);
            options.width = shape.first;
            options.height = shape.second;
        } else {
            throw std::invalid_argument("未知或缺少值的参数: " + argument);
        }
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const DemoOptions options = parse_options(argc, argv);

        // 先检查元素数量和 FP32 字节数，避免 Host 端容量计算溢出。
        const std::size_t element_count =
            cuda_foundations::test::checked_element_count(options.width, options.height);
        const std::size_t byte_count =
            cuda_foundations::test::checked_float_byte_count(element_count);

        // 两个版本使用完全相同的确定性输入；Reference 由所选 Kernel 决定。
        const std::vector<float> host_input =
            cuda_foundations::test::make_deterministic_input(element_count);
        const std::vector<float> host_reference =
            options.kernel_name == "copy"
                ? cuda_foundations::test::copy_reference(host_input)
                : cuda_foundations::test::transpose_reference(
                      host_input, options.width, options.height);
        std::vector<float> host_output(element_count, 0.0F);

        // Device 输入和输出都是相同元素数量的连续 FP32 Buffer。
        float* device_input = nullptr;
        float* device_output = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), byte_count));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), byte_count));

        // H2D 在 Kernel 前完成，不与演示程序的 Kernel 执行重叠。
        CUDA_CHECK(cudaMemcpy(
            device_input, host_input.data(), byte_count, cudaMemcpyHostToDevice));

        // 只根据 CLI 选择启动函数，其余分配、传输和验证路径完全一致。
        if (options.kernel_name == "copy") {
            cuda_foundations::transpose::launch_copy(
                device_input, device_output, options.width, options.height);
        } else {
            cuda_foundations::transpose::launch_naive(
                device_input, device_output, options.width, options.height);
        }

        // 演示路径显式等待 Kernel 完成，以捕获异步执行错误。
        CUDA_CHECK(cudaDeviceSynchronize());

        // D2H 完成后才能在 Host 端读取并验证输出。
        CUDA_CHECK(cudaMemcpy(
            host_output.data(), device_output, byte_count, cudaMemcpyDeviceToHost));

        std::string error_message;
        const bool correct = cuda_foundations::test::bitwise_equal(
            host_output, host_reference, &error_message);

        // 无论验证是否通过，都先按所有权释放两块 Device Buffer。
        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_input));

        if (!correct) {
            std::cerr << options.kernel_name << " 验证失败: " << error_message << std::endl;
            return EXIT_FAILURE;
        }

        // Copy 输出 Shape 不变；Naive 输出的逻辑 Shape 为 height × width。
        const std::size_t output_width =
            options.kernel_name == "copy" ? options.width : options.height;
        const std::size_t output_height =
            options.kernel_name == "copy" ? options.height : options.width;
        const std::size_t print_count = std::min<std::size_t>(element_count, 8U);

        std::cout << "Kernel " << options.kernel_name << " 验证通过，输入 Shape="
                  << options.width << 'x' << options.height << "，输出 Shape="
                  << output_width << 'x' << output_height << "\n前 " << print_count
                  << " 个输出:";
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
