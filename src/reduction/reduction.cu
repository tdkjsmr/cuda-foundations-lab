#include "cuda_check.cuh"
#include "reduction.cuh"
#include "reduction_utils.h"
#include "test_utils.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_foundations::reduction {

// V0 每个 Block 把最多 256 个输入元素归约成一个 Partial Sum。
__global__ void interleaved_reduction_kernel(const float* input,
                                             float* partial_sums,
                                             std::size_t input_count) {
    // 每个线程在 Shared Memory 中拥有一个固定槽位。
    __shared__ float shared_values[kBlockSize];

    // thread_index 是 Block 内线程号，global_index 是本阶段输入中的线性位置。
    const unsigned int thread_index = threadIdx.x;
    const std::size_t global_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + thread_index;

    // 超出当前阶段输入范围的线程写入 0，保证后续所有 Shared Memory 读取都合法。
    shared_values[thread_index] =
        global_index < input_count ? input[global_index] : 0.0F;

    // 所有 256 个槽位初始化完成后才能开始第一轮相加。
    __syncthreads();

    // stride 从 1 倍增到 128；每轮把间隔 stride 的右侧值加到左侧。
    for (unsigned int stride = 1U; stride < blockDim.x; stride *= 2U) {
        // V0 使用取模判断，活跃线程在 Warp 中呈交错分布。
        if (thread_index % (2U * stride) == 0U) {
            shared_values[thread_index] += shared_values[thread_index + stride];
        }

        // 下一轮依赖本轮写入；所有线程必须无条件到达 Block 屏障。
        __syncthreads();
    }

    // Block 的完整和最终位于 shared_values[0]，只由线程 0 写出。
    if (thread_index == 0U) {
        partial_sums[blockIdx.x] = shared_values[0];
    }
}

std::size_t workspace_elements(std::size_t input_count) {
    if (input_count == 0U) {
        throw std::invalid_argument("Reduction 输入元素数必须大于 0");
    }

    // 使用 1 + (N - 1) / block 避免 N + block - 1 发生 size_t 溢出。
    return 1U + (input_count - 1U) / kBlockSize;
}

std::size_t interleaved_launch_count(std::size_t input_count) {
    if (input_count == 0U) {
        throw std::invalid_argument("Reduction 输入元素数必须大于 0");
    }

    std::size_t launch_count = 0U;
    std::size_t current_count = input_count;

    // N=1 也执行一次 Kernel，保持所有输入规模都有统一的 Device 归约路径。
    do {
        current_count = workspace_elements(current_count);
        ++launch_count;
    } while (current_count > 1U);

    return launch_count;
}

// 启动一个阶段，并立即检查 Launch 配置错误；执行错误由调用方同步点捕获。
void launch_interleaved_stage(const float* device_input,
                              float* device_output,
                              std::size_t input_count,
                              cudaStream_t stream) {
    const std::size_t partial_count = workspace_elements(input_count);

    // Grid.x 使用 unsigned int；拒绝无法由 dim3 表示的极端输入。
    if (partial_count > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("Reduction 第一维 Grid 超出 dim3 表示范围");
    }

    const dim3 block(kBlockSize, 1U, 1U);
    const dim3 grid(static_cast<unsigned int>(partial_count), 1U, 1U);

    // Shared Memory 在 Kernel 内静态声明，动态 Shared Memory 字节数为 0。
    interleaved_reduction_kernel<<<grid, block, 0U, stream>>>(
        device_input, device_output, input_count);
    CUDA_CHECK(cudaGetLastError());
}

ReductionLaunchInfo reduce_interleaved(const float* device_input,
                                       float* workspace_a,
                                       float* workspace_b,
                                       std::size_t input_count,
                                       cudaStream_t stream) {
    // 输入、两块 Ping-Pong Workspace 必须独立且有效。
    if (device_input == nullptr || workspace_a == nullptr || workspace_b == nullptr ||
        device_input == workspace_a || device_input == workspace_b ||
        workspace_a == workspace_b) {
        throw std::invalid_argument("Reduction Device 指针为空或 Workspace 发生别名");
    }
    if (input_count == 0U) {
        throw std::invalid_argument("Reduction 输入元素数必须大于 0");
    }

    const std::size_t first_stage_partial_count = workspace_elements(input_count);
    const float* current_input = device_input;
    float* current_output = workspace_a;
    std::size_t current_count = input_count;
    std::size_t launch_count = 0U;

    // 每个阶段把 current_count 缩小为 ceil(current_count / 256)。
    do {
        launch_interleaved_stage(current_input, current_output, current_count, stream);
        current_count = workspace_elements(current_count);
        ++launch_count;

        // 当前输出只剩一个元素时，它就是最终 Device 结果。
        if (current_count == 1U) {
            return ReductionLaunchInfo{
                current_output, first_stage_partial_count, launch_count};
        }

        // 下一阶段读取当前输出，并写到另一块 Workspace，避免原地覆盖未读数据。
        current_input = current_output;
        current_output = current_output == workspace_a ? workspace_b : workspace_a;
    } while (true);
}

}  // namespace cuda_foundations::reduction

#ifndef CUDA_FOUNDATIONS_REDUCTION_CORE_ONLY

namespace {

// 演示程序默认使用一百万零三个随机元素，覆盖非 2 的幂和多阶段归约。
struct DemoOptions {
    std::size_t input_count = 1000003U;
    cuda_foundations::reduction::InputPattern pattern =
        cuda_foundations::reduction::InputPattern::kRandom;
};

// 把十进制 N 完整解析为非零 size_t。
std::size_t parse_size(const std::string& text) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument("--size 必须是 size_t 范围内的正整数");
    }
    std::size_t parsed_count = 0U;
    const unsigned long long parsed = std::stoull(text, &parsed_count);
    if (parsed_count != text.size() || parsed == 0ULL ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("--size 必须是 size_t 范围内的正整数");
    }
    return static_cast<std::size_t>(parsed);
}

// 解析 --size N 与 --pattern NAME，参数顺序可以互换。
DemoOptions parse_options(int argc, char** argv) {
    DemoOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--size" && index + 1 < argc) {
            options.input_count = parse_size(argv[++index]);
        } else if (argument == "--pattern" && index + 1 < argc) {
            options.pattern = cuda_foundations::reduction::parse_pattern(argv[++index]);
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

        // Host 输入和 double Reference 在任何 Device 分配前准备完成。
        const std::vector<float> host_input =
            cuda_foundations::reduction::make_input(options.input_count, options.pattern);
        const cuda_foundations::reduction::CpuReference reference =
            cuda_foundations::reduction::cpu_reference(host_input);

        // 输入 Buffer 按 N 分配；两块 Workspace 只需容纳第一阶段 Partial Sums。
        const std::size_t input_bytes =
            cuda_foundations::test::checked_float_byte_count(options.input_count);
        const std::size_t workspace_count =
            cuda_foundations::reduction::workspace_elements(options.input_count);
        const std::size_t workspace_bytes =
            cuda_foundations::test::checked_float_byte_count(workspace_count);

        float* device_input = nullptr;
        float* workspace_a = nullptr;
        float* workspace_b = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), input_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace_a), workspace_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace_b), workspace_bytes));

        // H2D 在归约前完成，不计入 Kernel-only 性能路径。
        CUDA_CHECK(cudaMemcpy(
            device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));

        // Host 只调度阶段；Partial Sums 始终保留在 Device Workspace 中。
        const cuda_foundations::reduction::ReductionLaunchInfo launch_info =
            cuda_foundations::reduction::reduce_interleaved(
                device_input, workspace_a, workspace_b, options.input_count);

        // 演示路径显式同步，以捕获多阶段中的异步执行错误。
        CUDA_CHECK(cudaDeviceSynchronize());

        float gpu_result = 0.0F;
        CUDA_CHECK(cudaMemcpy(
            &gpu_result, launch_info.device_result, sizeof(float), cudaMemcpyDeviceToHost));

        const cuda_foundations::reduction::ErrorMetrics errors =
            cuda_foundations::reduction::error_metrics(gpu_result, reference);

        // 按分配逆序释放三个 Device Buffer。
        CUDA_CHECK(cudaFree(workspace_b));
        CUDA_CHECK(cudaFree(workspace_a));
        CUDA_CHECK(cudaFree(device_input));

        std::cout << "Kernel: interleaved_v0\n"
                  << "N: " << options.input_count << "\n"
                  << "Pattern: "
                  << cuda_foundations::reduction::pattern_name(options.pattern) << "\n"
                  << "Block: " << cuda_foundations::reduction::kBlockSize << "\n"
                  << "First-stage partials: " << launch_info.first_stage_partial_count
                  << "\nKernel launches: " << launch_info.launch_count << "\n"
                  << "GPU result: " << gpu_result << "\n"
                  << "CPU double reference: " << reference.sum << "\n"
                  << "Absolute error: " << errors.absolute_error << "\n"
                  << "Normalized error: " << errors.normalized_error << "\n"
                  << "Tolerance: " << errors.tolerance << std::endl;

        if (errors.absolute_error > errors.tolerance) {
            std::cerr << "Reduction V0 误差超过阈值" << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Correctness: PASS" << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "程序失败: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}

#endif
