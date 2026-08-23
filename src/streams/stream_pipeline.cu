#include "cuda_check.cuh"
#include "stream_pipeline.cuh"
#include "test_utils.h"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

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

namespace {

// 统一检查 size_t 乘法，防止 Host/Device Buffer 容量静默回绕。
std::size_t checked_multiply(std::size_t left,
                             std::size_t right,
                             const char* error_message) {
    // 任一因子为 0 时结果可以直接返回，不做除法。
    if (left == 0U || right == 0U) {
        return 0U;
    }

    // left > max / right 等价于 left * right 不可由 size_t 表示。
    if (left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(error_message);
    }

    // 只在已证明不溢出后才执行乘法。
    return left * right;
}

// 根据开关插入 NVTX Push；正式 Benchmark 默认不支付这项标注开销。
void begin_nvtx_range(const char* name, bool enabled) {
    if (enabled) {
        nvtxRangePushA(name);
    }
}

// 与 begin_nvtx_range 成对，仅 Profile 模式实际调用 NVTX。
void end_nvtx_range(bool enabled) {
    if (enabled) {
        nvtxRangePop();
    }
}

}  // namespace

namespace cuda_foundations::streams {

// Stream 子项目冻结复用 Transpose V3 的 Padded Shared-Memory 算法。
__global__ void stream_padded_transpose_kernel(const float* input,
                                               float* output,
                                               std::size_t width,
                                               std::size_t height) {
    // 32 x 33 的行跨度让 Warp 按列读取时轮换 Shared Memory Bank。
    __shared__ float tile[kTileDim][kPaddedTileColumns];

    // blockIdx.x 选择输入 Tile 的横向坐标，threadIdx.x 选择列。
    const std::size_t input_x =
        static_cast<std::size_t>(blockIdx.x) * kTileDim + threadIdx.x;

    // blockIdx.y 选择输入 Tile 的纵向坐标，threadIdx.y 选择首行。
    const std::size_t input_base_y =
        static_cast<std::size_t>(blockIdx.y) * kTileDim + threadIdx.y;

    // 256 个线程每个加载 4 个元素，共同覆盖一个 32 x 32 Tile。
    for (unsigned int row_offset = 0U;
         row_offset < kTileDim;
         row_offset += kBlockRows) {
        // 同一线程处理的四行之间相隔 kBlockRows。
        const std::size_t input_y = input_base_y + row_offset;

        // 最后一个不完整 Tile 中只读取原矩阵内部的元素。
        if (input_x < width && input_y < height) {
            // Warp 内相邻 threadIdx.x 执行连续 Global Load。
            tile[threadIdx.y + row_offset][threadIdx.x] =
                input[input_y * width + input_x];
        }
    }

    // 交换 Shared Memory 行列索引前，所有线程必须看到完整 Tile。
    __syncthreads();

    // 交换 blockIdx.x/y 的角色，使相邻线程连续写转置后的列。
    const std::size_t output_x =
        static_cast<std::size_t>(blockIdx.y) * kTileDim + threadIdx.x;

    // 转置输出的逻辑 Shape 为 height x width。
    const std::size_t output_base_y =
        static_cast<std::size_t>(blockIdx.x) * kTileDim + threadIdx.y;

    // 以与加载阶段相同的 8 行步长写回四个输出元素。
    for (unsigned int row_offset = 0U;
         row_offset < kTileDim;
         row_offset += kBlockRows) {
        // output_y 是转置后矩阵的行坐标。
        const std::size_t output_y = output_base_y + row_offset;

        // 输出边界分别对应原输入的 height 和 width。
        if (output_x < height && output_y < width) {
            // 33-word 行跨度避免列式 Shared Load 的 32-way Bank Conflict。
            output[output_y * height + output_x] =
                tile[threadIdx.x][threadIdx.y + row_offset];
        }
    }
}

WorkloadLayout validate_and_derive_layout(const StreamWorkload& workload) {
    // 零尺寸不能形成有效矩阵或 CUDA Grid。
    if (workload.width == 0U || workload.height == 0U) {
        throw std::invalid_argument("Stream 矩阵的 width 和 height 必须大于 0");
    }

    // 零 Chunk 不包含可测量的 Pipeline 工作。
    if (workload.chunk_count == 0U) {
        throw std::invalid_argument("Stream Pipeline 的 chunk_count 必须大于 0");
    }

    // 第一次乘法得到每个独立矩阵 Chunk 的 FP32 元素数。
    const std::size_t elements_per_chunk = checked_multiply(
        workload.width, workload.height, "Stream 单 Chunk 元素数发生 size_t 溢出");

    // 第二次乘法将元素数换算为精确字节数。
    const std::size_t bytes_per_chunk = checked_multiply(
        elements_per_chunk, sizeof(float), "Stream 单 Chunk 字节数发生 size_t 溢出");

    // 第三次乘法得到整批 Host Input/Output 各自的元素数。
    const std::size_t total_elements = checked_multiply(
        elements_per_chunk,
        workload.chunk_count,
        "Stream 整批 Chunk 元素数发生 size_t 溢出");

    // 最后用已验证的单 Chunk 字节数计算整批 Payload。
    const std::size_t total_bytes = checked_multiply(
        bytes_per_chunk,
        workload.chunk_count,
        "Stream 整批 Chunk 字节数发生 size_t 溢出");

    // 容量算术全部验证后，再检查 CUDA 8.6 目标的 Grid X/Y 上限。
    const std::size_t grid_x = 1U + (workload.width - 1U) / kTileDim;
    const std::size_t grid_y = 1U + (workload.height - 1U) / kTileDim;
    if (grid_x > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        grid_y > 65535U) {
        throw std::overflow_error("Stream Transpose Grid 超出 CUDA Device 可表示范围");
    }

    // 所有容量都在返回前完成了非零和溢出验证。
    return WorkloadLayout{
        elements_per_chunk, bytes_per_chunk, total_elements, total_bytes};
}

DeviceCapabilities query_device_capabilities() {
    // 首先读取当前 Runtime Device，避免假定一定使用 Device 0。
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));

    // cudaGetDeviceProperties 一次性提供设备名称和三个并发能力字段。
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));

    // 能力为 0 时 V3.0 同步基线仍然可以正常运行。
    return DeviceCapabilities{
        device_id,
        properties.name,
        properties.deviceOverlap,
        properties.asyncEngineCount,
        properties.concurrentKernels};
}

DeviceBufferPair::DeviceBufferPair(DeviceBufferPair&& other) noexcept
    : input(other.input),
      output(other.output),
      capacity_bytes(other.capacity_bytes) {
    // 转移所有权后立即清空源对象，保证其析构不会重复释放 Device 指针。
    other.input = nullptr;
    other.output = nullptr;
    other.capacity_bytes = 0U;
}

DeviceBufferPair& DeviceBufferPair::operator=(DeviceBufferPair&& other) noexcept {
    if (this != &other) {
        // 先释放目标对象原有资源，再接管源对象，避免 Move Assignment 泄漏。
        release_device_buffers(this);
        input = other.input;
        output = other.output;
        capacity_bytes = other.capacity_bytes;
        other.input = nullptr;
        other.output = nullptr;
        other.capacity_bytes = 0U;
    }
    return *this;
}

DeviceBufferPair::~DeviceBufferPair() {
    // 显式 release 后指针已经清空；遗漏显式清理时析构仍会提供兜底回收。
    release_device_buffers(this);
}

DeviceBufferPair allocate_device_buffers(std::size_t capacity_bytes) {
    // cudaMalloc(0) 的语义不适合本项目，因此显式拒绝零容量。
    if (capacity_bytes == 0U) {
        throw std::invalid_argument("Device Buffer 容量必须大于 0");
    }

    // 先将所有者结构保持为空，再逐块填入成功的分配。
    DeviceBufferPair buffers{};
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.input), capacity_bytes));
    const cudaError_t output_status =
        cudaMalloc(reinterpret_cast<void**>(&buffers.output), capacity_bytes);
    if (output_status != cudaSuccess) {
        // 第二块分配失败时先回收第一块，避免 fatal 错误路径遗留半套所有权。
        CUDA_CHECK(cudaFree(buffers.input));
        buffers.input = nullptr;
        CUDA_CHECK(output_status);
    }
    buffers.capacity_bytes = capacity_bytes;
    return buffers;
}

void release_device_buffers(DeviceBufferPair* buffers) {
    // 所有者结构自身不能为空，否则无法清理其状态。
    if (buffers == nullptr) {
        throw std::invalid_argument("release_device_buffers 收到空所有者指针");
    }

    // 按与分配相反的顺序先释放 Output。
    if (buffers->output != nullptr) {
        CUDA_CHECK(cudaFree(buffers->output));
        buffers->output = nullptr;
    }

    // 再释放 Input，使这对 Buffer 不再拥有任何 Device 资源。
    if (buffers->input != nullptr) {
        CUDA_CHECK(cudaFree(buffers->input));
        buffers->input = nullptr;
    }

    // 指针全部清空后同时清除容量，防止误认为仍可复用。
    buffers->capacity_bytes = 0U;
}

void launch_padded_tiled_transpose(const float* input,
                                  float* output,
                                  std::size_t width,
                                  std::size_t height,
                                  cudaStream_t stream) {
    // Kernel 不接受空 Device 指针或零尺寸矩阵。
    if (input == nullptr || output == nullptr || width == 0U || height == 0U) {
        throw std::invalid_argument("Stream Transpose Launcher 收到无效参数");
    }

    // 使用 1 + (n - 1) / tile 向上取整，避免 n + tile - 1 溢出。
    const std::size_t grid_x = 1U + (width - 1U) / kTileDim;
    const std::size_t grid_y = 1U + (height - 1U) / kTileDim;
    if (grid_x > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        grid_y > 65535U) {
        throw std::overflow_error("Stream Transpose Launcher 的 Grid 超出设备范围");
    }

    // Block 保持 Transpose V3 的 32 x 8 = 256 线程配置。
    const dim3 block(kTileDim, kBlockRows, 1U);
    const dim3 grid(
        static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y), 1U);

    // V3.0 传入 nullptr，因此 Kernel 位于默认 Stream；V3.2 将显式传入非默认 Stream。
    stream_padded_transpose_kernel<<<grid, block, 0U, stream>>>(
        input, output, width, height);

    // 只检查 Launch 的立即错误，执行阶段错误由后续同步拷贝或显式同步捕获。
    CUDA_CHECK(cudaGetLastError());
}

void execute_synchronous_pipeline(const StreamWorkload& workload,
                                  const WorkloadLayout& layout,
                                  const float* host_input,
                                  float* host_output,
                                  const DeviceBufferPair& buffers,
                                  bool annotate_nvtx) {
    // 重新派生并逐字段核对，防止公共执行入口收到伪造或过期的 Buffer 容量。
    const WorkloadLayout expected_layout = validate_and_derive_layout(workload);
    if (layout.elements_per_chunk != expected_layout.elements_per_chunk ||
        layout.bytes_per_chunk != expected_layout.bytes_per_chunk ||
        layout.total_elements != expected_layout.total_elements ||
        layout.total_bytes != expected_layout.total_bytes) {
        throw std::invalid_argument("StreamWorkload 与 WorkloadLayout 不一致");
    }

    // Pageable/Pinned Host Buffer 和 Device Buffer 都必须在进入 Pipeline 前有效。
    if (host_input == nullptr || host_output == nullptr || buffers.input == nullptr ||
        buffers.output == nullptr) {
        throw std::invalid_argument("Stream Pipeline 收到空 Host/Device Buffer");
    }

    // 同步基线只复用一对 Device Buffer，容量必须容纳完整 Chunk。
    if (buffers.capacity_bytes < layout.bytes_per_chunk) {
        throw std::invalid_argument("Stream Device Buffer 容量小于单 Chunk");
    }

    // 外层范围表示一次完整 Pipeline Iteration。
    begin_nvtx_range("pipeline_iteration", annotate_nvtx);

    // 每个 Chunk 是一张独立矩阵，Host Input/Output 使用相同的 Chunk 偏移。
    for (std::size_t chunk_index = 0U;
         chunk_index < workload.chunk_count;
         ++chunk_index) {
        // Profile 模式为每个 Chunk 生成唯一标签，正式计时不构造字符串。
        const std::string chunk_label =
            annotate_nvtx ? "chunk_" + std::to_string(chunk_index) : std::string{};
        begin_nvtx_range(chunk_label.c_str(), annotate_nvtx);

        // 乘法已在 total_elements 校验中证明不溢出。
        const std::size_t chunk_offset = chunk_index * layout.elements_per_chunk;
        const float* host_chunk_input = host_input + chunk_offset;
        float* host_chunk_output = host_output + chunk_offset;

        // Baseline A 必须使用 blocking cudaMemcpy 和 Pageable Host Memory。
        begin_nvtx_range("h2d", annotate_nvtx);
        CUDA_CHECK(cudaMemcpy(buffers.input,
                              host_chunk_input,
                              layout.bytes_per_chunk,
                              cudaMemcpyHostToDevice));
        end_nvtx_range(annotate_nvtx);

        // Kernel 与已完成的 Padded Tiled Transpose 保持相同算法和启动形状。
        begin_nvtx_range("transpose", annotate_nvtx);
        launch_padded_tiled_transpose(
            buffers.input, buffers.output, workload.width, workload.height, nullptr);
        end_nvtx_range(annotate_nvtx);

        // blocking D2H 只有在前面的默认 Stream Kernel 完成后才返回。
        begin_nvtx_range("d2h", annotate_nvtx);
        CUDA_CHECK(cudaMemcpy(host_chunk_output,
                              buffers.output,
                              layout.bytes_per_chunk,
                              cudaMemcpyDeviceToHost));
        end_nvtx_range(annotate_nvtx);

        // 同步 D2H 返回后才能安全复用同一对 Device Buffer 处理下一 Chunk。
        end_nvtx_range(annotate_nvtx);
    }

    // 调用方在计时边界执行 cudaDeviceSynchronize，因此此处不添加额外同步。
    end_nvtx_range(annotate_nvtx);
}

}  // namespace cuda_foundations::streams

#ifndef CUDA_FOUNDATIONS_STREAM_CORE_ONLY

namespace {

// 演示程序默认使用较小工作量，正式 512 MiB 实验由 Benchmark 脚本运行。
struct DemoOptions {
    cuda_foundations::streams::StreamWorkload workload{1024U, 1024U, 4U};
};

// 将 WIDTHxHEIGHT 严格解析为两个非零 size_t。
std::pair<std::size_t, std::size_t> parse_shape(const std::string& text) {
    const std::size_t separator = text.find('x');
    if (separator == std::string::npos || separator == 0U ||
        separator + 1U >= text.size()) {
        throw std::invalid_argument("Shape 必须使用 WIDTHxHEIGHT 格式");
    }

    // 分别解析宽高，并记录 stoull 实际消费的字符数。
    const std::string width_text = text.substr(0U, separator);
    const std::string height_text = text.substr(separator + 1U);
    std::size_t width_characters = 0U;
    std::size_t height_characters = 0U;
    const unsigned long long parsed_width = std::stoull(width_text, &width_characters);
    const unsigned long long parsed_height = std::stoull(height_text, &height_characters);

    // 拒绝零、尾随字符和超出 size_t 范围的输入。
    if (width_characters != width_text.size() ||
        height_characters != height_text.size() || parsed_width == 0ULL ||
        parsed_height == 0ULL ||
        parsed_width > std::numeric_limits<std::size_t>::max() ||
        parsed_height > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("Shape 必须包含两个 size_t 范围内的正整数");
    }

    return {static_cast<std::size_t>(parsed_width),
            static_cast<std::size_t>(parsed_height)};
}

// 将 --chunks 严格解析为非零 size_t。
std::size_t parse_positive_size(const std::string& text, const char* option_name) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string(option_name) + " 必须是正整数");
    }

    std::size_t parsed_characters = 0U;
    const unsigned long long parsed = std::stoull(text, &parsed_characters);
    if (parsed_characters != text.size() || parsed == 0ULL ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(option_name) + " 必须是正整数");
    }
    return static_cast<std::size_t>(parsed);
}

// 演示程序只允许改变矩阵 Shape 和 Chunk 数。
DemoOptions parse_options(int argc, char** argv) {
    DemoOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--shape" && index + 1 < argc) {
            const auto [width, height] = parse_shape(argv[++index]);
            options.workload.width = width;
            options.workload.height = height;
        } else if (argument == "--chunks" && index + 1 < argc) {
            options.workload.chunk_count = parse_positive_size(argv[++index], "--chunks");
        } else {
            throw std::invalid_argument("未知或缺少值的参数: " + argument);
        }
    }
    return options;
}

// 为整批独立矩阵构造 CPU Transpose Reference。
std::vector<float> make_batch_reference(
    const std::vector<float>& input,
    const cuda_foundations::streams::StreamWorkload& workload,
    const cuda_foundations::streams::WorkloadLayout& layout) {
    // Reference 与 Host Output 使用相同的连续 Chunk 布局。
    std::vector<float> reference(layout.total_elements, 0.0F);

    for (std::size_t chunk = 0U; chunk < workload.chunk_count; ++chunk) {
        const std::size_t chunk_offset = chunk * layout.elements_per_chunk;
        for (std::size_t y = 0U; y < workload.height; ++y) {
            for (std::size_t x = 0U; x < workload.width; ++x) {
                // 每个 Chunk 都独立把 input[y][x] 写到 output[x][y]。
                reference[chunk_offset + x * workload.height + y] =
                    input[chunk_offset + y * workload.width + x];
            }
        }
    }
    return reference;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // 解析参数并在任何分配前完成全部容量校验。
        const DemoOptions options = parse_options(argc, argv);
        const cuda_foundations::streams::WorkloadLayout layout =
            cuda_foundations::streams::validate_and_derive_layout(options.workload);

        // std::vector 保证 V3.0 的 Host Buffer 是普通 Pageable Memory。
        std::vector<float> host_input(layout.total_elements, 0.0F);
        std::vector<float> host_output(layout.total_elements, 0.0F);

        // 把 Chunk 编号纳入模式，避免错误重复 Chunk 0 仍然通过验证。
        for (std::size_t chunk = 0U; chunk < options.workload.chunk_count; ++chunk) {
            const std::size_t chunk_offset = chunk * layout.elements_per_chunk;
            for (std::size_t element = 0U; element < layout.elements_per_chunk; ++element) {
                const std::size_t pattern = (element * 17U + chunk * 29U) % 509U;
                host_input[chunk_offset + element] =
                    static_cast<float>(static_cast<int>(pattern) - 254) * 0.125F;
            }
        }

        // CPU Reference 构造在 Pipeline 执行之前，不会混入后续性能计时。
        const std::vector<float> host_reference =
            make_batch_reference(host_input, options.workload, layout);

        // 同步基线仅需一对单 Chunk 容量的 Device Buffer。
        cuda_foundations::streams::DeviceBufferPair buffers =
            cuda_foundations::streams::allocate_device_buffers(layout.bytes_per_chunk);

        // 完整执行所有 blocking H2D -> Kernel -> blocking D2H。
        cuda_foundations::streams::execute_synchronous_pipeline(
            options.workload,
            layout,
            host_input.data(),
            host_output.data(),
            buffers,
            false);

        // 明确建立整批工作完成边界，同时捕获异步执行错误。
        CUDA_CHECK(cudaDeviceSynchronize());

        // Transpose 只重排位模式，因此对 NaN 也使用逐位验证。
        std::string error_message;
        const bool correct = cuda_foundations::test::bitwise_equal(
            host_output, host_reference, &error_message);

        // 无论验证是否通过，先释放本进程拥有的 Device Buffer。
        cuda_foundations::streams::release_device_buffers(&buffers);

        if (!correct) {
            std::cerr << "Stream V3.0 验证失败: " << error_message << std::endl;
            return EXIT_FAILURE;
        }

        // 能力字段与 NSYS 时间线共同用于后续并发验收。
        const cuda_foundations::streams::DeviceCapabilities capabilities =
            cuda_foundations::streams::query_device_capabilities();

        std::cout << "Pipeline: pageable_sync_v0\n"
                  << "Host memory: Pageable (std::vector)\n"
                  << "Copy API: cudaMemcpy (blocking)\n"
                  << "Kernel: stream_padded_transpose_kernel\n"
                  << "Shape/chunk: " << options.workload.width << 'x'
                  << options.workload.height << " FP32\n"
                  << "Chunks: " << options.workload.chunk_count << '\n'
                  << "Chunk MiB: "
                  << static_cast<double>(layout.bytes_per_chunk) / (1024.0 * 1024.0)
                  << '\n'
                  << "Total payload MiB: "
                  << static_cast<double>(layout.total_bytes) / (1024.0 * 1024.0)
                  << '\n'
                  << "GPU: " << capabilities.device_name << '\n'
                  << "deviceOverlap: " << capabilities.device_overlap << '\n'
                  << "asyncEngineCount: " << capabilities.async_engine_count << '\n'
                  << "concurrentKernels: " << capabilities.concurrent_kernels << '\n'
                  << "Bitwise correctness: PASS" << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        // CLI、容量或 Host 验证失败时输出具体原因。
        std::cerr << "Stream V3.0 程序失败: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}

#endif
