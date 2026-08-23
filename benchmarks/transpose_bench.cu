#include "benchmark.h"
#include "cuda_check.cuh"
#include "cuda_timer.cuh"
#include "test_utils.h"
#include "transpose.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef CUDA_FOUNDATIONS_GIT_COMMIT
#define CUDA_FOUNDATIONS_GIT_COMMIT "unknown"
#endif

namespace {

// 四个 Kernel 的 Host 启动函数签名一致，可共享同一套 Benchmark 流程。
using LaunchFunction = void (*)(const float*,
                                float*,
                                std::size_t,
                                std::size_t,
                                cudaStream_t);

// 集中保存 CLI 参数，保证日志能够完整复现实验配置。
struct Options {
    std::string kernel_name = "all";
    std::size_t width = 4096U;
    std::size_t height = 4096U;
    int warmup_count = 20;
    int iteration_count = 100;
    int group_count = 5;
    bool profile_mode = false;
    std::string csv_path;
};

// 描述一个待测 Kernel 的名字、启动函数以及正确的 CPU Reference 语义。
struct KernelSpec {
    const char* name;
    LaunchFunction launch;
    bool transpose_output;
};

// 保存一个 Kernel 的统计结果与由中位数计算的有效带宽。
struct KernelResult {
    KernelSpec spec;
    cuda_foundations::benchmark::Summary timing_us;
    double bandwidth_gbps;
    double relative_copy_percent;
    double relative_naive_speedup;
};

// 解析 WIDTHxHEIGHT，并确认两个数字都被完整消费。
std::pair<std::size_t, std::size_t> parse_shape(const std::string& text) {
    const std::size_t separator = text.find('x');
    if (separator == std::string::npos || separator == 0U || separator + 1U >= text.size()) {
        throw std::invalid_argument("--shape 必须使用 WIDTHxHEIGHT 格式");
    }

    const std::string width_text = text.substr(0U, separator);
    const std::string height_text = text.substr(separator + 1U);
    std::size_t width_characters = 0U;
    std::size_t height_characters = 0U;
    const std::size_t width = std::stoull(width_text, &width_characters);
    const std::size_t height = std::stoull(height_text, &height_characters);

    if (width_characters != width_text.size() || height_characters != height_text.size()) {
        throw std::invalid_argument("--shape 包含无法解析的尾随字符");
    }

    cuda_foundations::test::checked_element_count(width, height);
    return {width, height};
}

// 将十进制文本转换为正整数，拒绝零、负数和尾随字符。
int parse_positive_int(const std::string& text, const std::string& option_name) {
    std::size_t parsed_count = 0U;
    const int value = std::stoi(text, &parsed_count);
    if (parsed_count != text.size() || value <= 0) {
        throw std::invalid_argument(option_name + " 必须是正整数");
    }
    return value;
}

// 解析 Kernel、Shape、预热、迭代、组数、Profile 和 CSV 参数。
Options parse_options(int argc, char** argv) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "--kernel" && index + 1 < argc) {
            options.kernel_name = argv[++index];
            if (options.kernel_name != "copy" && options.kernel_name != "naive" &&
                options.kernel_name != "tiled" && options.kernel_name != "padded" &&
                options.kernel_name != "all") {
                throw std::invalid_argument("--kernel 只支持 copy、naive、tiled、padded 或 all");
            }
        } else if (argument == "--shape" && index + 1 < argc) {
            const auto shape = parse_shape(argv[++index]);
            options.width = shape.first;
            options.height = shape.second;
        } else if (argument == "--warmup" && index + 1 < argc) {
            options.warmup_count = parse_positive_int(argv[++index], "--warmup");
        } else if (argument == "--iterations" && index + 1 < argc) {
            options.iteration_count = parse_positive_int(argv[++index], "--iterations");
        } else if (argument == "--groups" && index + 1 < argc) {
            options.group_count = parse_positive_int(argv[++index], "--groups");
        } else if (argument == "--csv" && index + 1 < argc) {
            options.csv_path = argv[++index];
        } else if (argument == "--profile") {
            options.profile_mode = true;
        } else {
            throw std::invalid_argument("未知或缺少值的参数: " + argument);
        }
    }

    // Profile 模式只允许选择一个 Kernel，避免时间线或报告混入另一版本。
    if (options.profile_mode && options.kernel_name == "all") {
        throw std::invalid_argument("--profile 必须配合 --kernel copy、naive、tiled 或 padded");
    }

    // Profile 模式在预热后只留下一个正式 Launch，减少报告中的重复事件。
    if (options.profile_mode) {
        options.iteration_count = 1;
        options.group_count = 1;
    }

    return options;
}

// 根据 --kernel 生成有确定顺序的待测列表；all 始终按 Copy、Naive、Tiled、Padded 顺序测量。
std::vector<KernelSpec> selected_kernels(const std::string& kernel_name) {
    const KernelSpec copy{
        "copy_v0", cuda_foundations::transpose::launch_copy, false};
    const KernelSpec naive{
        "naive_v1", cuda_foundations::transpose::launch_naive, true};
    const KernelSpec tiled{
        "tiled_v2", cuda_foundations::transpose::launch_tiled, true};
    const KernelSpec padded{
        "padded_v3", cuda_foundations::transpose::launch_padded, true};

    if (kernel_name == "copy") {
        return {copy};
    }
    if (kernel_name == "naive") {
        return {naive};
    }
    if (kernel_name == "tiled") {
        return {tiled};
    }
    if (kernel_name == "padded") {
        return {padded};
    }
    return {copy, naive, tiled, padded};
}

// 生成 UTC ISO-8601 时间戳，避免不同服务器时区导致 CSV 难以对齐。
std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc_time{};
    gmtime_r(&now, &utc_time);

    std::ostringstream stream;
    stream << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

// 把 CUDA 整数版本号转换成 major.minor 文本，例如 12040 → 12.4。
std::string format_cuda_version(int version) {
    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

// CSV 字段包含逗号或双引号时使用 RFC 4180 风格转义。
std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";
    for (const char character : value) {
        escaped += character == '\"' ? "\"\"" : std::string(1U, character);
    }
    escaped += '\"';
    return escaped;
}

// 为当前 Kernel 建立正确的 CPU Reference。
std::vector<float> make_reference(const KernelSpec& spec,
                                  const std::vector<float>& host_input,
                                  std::size_t width,
                                  std::size_t height) {
    if (spec.transpose_output) {
        return cuda_foundations::test::transpose_reference(host_input, width, height);
    }
    return cuda_foundations::test::copy_reference(host_input);
}

// 对单个 Kernel 执行预热、5 组 Event 计时和计时后的按位正确性验证。
KernelResult benchmark_kernel(const KernelSpec& spec,
                              const Options& options,
                              const std::vector<float>& host_input,
                              float* device_input,
                              float* device_output,
                              std::size_t byte_count) {
    // 每个版本都独立预热，避免 Copy 的热身状态被直接当作 Naive 的热身。
    for (int iteration = 0; iteration < options.warmup_count; ++iteration) {
        spec.launch(
            device_input, device_output, options.width, options.height, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_foundations::CudaEventTimer timer;
    std::vector<double> group_average_us;
    group_average_us.reserve(static_cast<std::size_t>(options.group_count));

    // 每组只在边界记录 Event，组内连续 Launch，不插入逐轮同步。
    for (int group = 0; group < options.group_count; ++group) {
        timer.start();
        for (int iteration = 0; iteration < options.iteration_count; ++iteration) {
            spec.launch(
                device_input, device_output, options.width, options.height, nullptr);
        }
        const float total_ms = timer.stop();
        const double average_us =
            static_cast<double>(total_ms) * 1000.0 /
            static_cast<double>(options.iteration_count);
        group_average_us.push_back(average_us);
    }

    // 正式计时后才执行 D2H 和 CPU 比较，不把验证开销混入 Kernel 时间。
    std::vector<float> host_output(host_input.size(), 0.0F);
    CUDA_CHECK(cudaMemcpy(
        host_output.data(), device_output, byte_count, cudaMemcpyDeviceToHost));
    const std::vector<float> reference =
        make_reference(spec, host_input, options.width, options.height);

    std::string error_message;
    if (!cuda_foundations::test::bitwise_equal(
            host_output, reference, &error_message)) {
        throw std::runtime_error(
            std::string(spec.name) + " 正确性验证失败: " + error_message);
    }

    const cuda_foundations::benchmark::Summary summary =
        cuda_foundations::benchmark::summarize(group_average_us);

    // 四个版本都读取并写入同样数量的 FP32 元素，因此使用相同有效字节数定义。
    const double transferred_bytes = 2.0 * static_cast<double>(byte_count);
    const double bandwidth_gbps =
        transferred_bytes / (summary.median * 1.0e-6) / 1.0e9;

    return KernelResult{spec, summary, bandwidth_gbps, 0.0, 0.0};
}

// 在同时测得 V0/V1/V2/V3 后计算相对 Copy 带宽比例和相对 Naive 加速比。
void compute_relative_metrics(std::vector<KernelResult>* results) {
    const KernelResult* copy_result = nullptr;
    const KernelResult* naive_result = nullptr;

    for (const KernelResult& result : *results) {
        if (std::string(result.spec.name) == "copy_v0") {
            copy_result = &result;
        } else if (std::string(result.spec.name) == "naive_v1") {
            naive_result = &result;
        }
    }

    for (KernelResult& result : *results) {
        if (copy_result != nullptr) {
            result.relative_copy_percent =
                result.bandwidth_gbps / copy_result->bandwidth_gbps * 100.0;
        }
        if (naive_result != nullptr) {
            result.relative_naive_speedup =
                naive_result->timing_us.median / result.timing_us.median;
        }
    }
}

// 打印单个 Kernel 的绝对指标和同负载相对指标。
void print_result(const KernelResult& result,
                  const Options& options,
                  const dim3& grid,
                  const dim3& block) {
    std::cout << std::fixed << std::setprecision(3)
              << "Kernel: " << result.spec.name << "\n"
              << "Input Shape: " << options.width << 'x' << options.height << " FP32\n"
              << "Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n"
              << "Block: (" << block.x << ", " << block.y << ", " << block.z << ")\n"
              << "Warm-up: " << options.warmup_count << "\n"
              << "Iterations/group: " << options.iteration_count << "\n"
              << "Groups: " << options.group_count << "\n"
              << "Min: " << result.timing_us.minimum << " us\n"
              << "Median (P50): " << result.timing_us.median << " us\n"
              << "P95: " << result.timing_us.p95 << " us\n"
              << "Stddev: " << result.timing_us.standard_deviation << " us\n"
              << "Effective bandwidth: " << result.bandwidth_gbps << " GB/s\n";

    if (result.relative_copy_percent > 0.0) {
        std::cout << "Relative to Copy bandwidth: " << result.relative_copy_percent
                  << "%\n";
    }
    if (result.relative_naive_speedup > 0.0) {
        std::cout << "Speedup relative to Naive: " << result.relative_naive_speedup
                  << "x\n";
    }

    std::cout << "Bitwise correctness: PASS\n" << std::endl;
}

// 将一个 Kernel 的实验结果追加到 CSV；首次创建文件时写固定表头。
void append_csv(const std::string& path,
                const Options& options,
                const KernelResult& result,
                const cudaDeviceProp& properties,
                int driver_version,
                int runtime_version,
                const dim3& grid,
                const dim3& block) {
    const std::filesystem::path csv_path(path);
    if (csv_path.has_parent_path()) {
        std::filesystem::create_directories(csv_path.parent_path());
    }

    const bool write_header =
        !std::filesystem::exists(csv_path) || std::filesystem::file_size(csv_path) == 0U;
    std::ofstream output(csv_path, std::ios::app);
    if (!output) {
        throw std::runtime_error("无法打开 CSV: " + path);
    }

    if (write_header) {
        output << "git_commit,timestamp,gpu_name,driver_version,cuda_version,"
                  "kernel_name,shape,dtype,grid,block,warmup_count,iteration_count,"
                  "group_count,min_us,median_us,p95_us,stddev_us,"
                  "effective_bandwidth_gbps,relative_copy_percent,"
                  "relative_naive_speedup,max_abs_error,relative_error\n";
    }

    output << CUDA_FOUNDATIONS_GIT_COMMIT << ','
           << utc_timestamp() << ','
           << csv_escape(properties.name) << ','
           << format_cuda_version(driver_version) << ','
           << format_cuda_version(runtime_version) << ','
           << result.spec.name << ',' << options.width << 'x' << options.height << ",FP32,"
           << csv_escape(std::to_string(grid.x) + "x" + std::to_string(grid.y)) << ','
           << csv_escape(std::to_string(block.x) + "x" + std::to_string(block.y)) << ','
           << options.warmup_count << ',' << options.iteration_count << ','
           << options.group_count << ',' << result.timing_us.minimum << ','
           << result.timing_us.median << ',' << result.timing_us.p95 << ','
           << result.timing_us.standard_deviation << ',' << result.bandwidth_gbps << ','
           << result.relative_copy_percent << ',' << result.relative_naive_speedup
           << ",0,0\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::size_t element_count =
            cuda_foundations::test::checked_element_count(options.width, options.height);
        const std::size_t byte_count =
            cuda_foundations::test::checked_float_byte_count(element_count);

        // 查询设备、驱动和 Runtime 版本，确保每条性能记录带完整环境信息。
        int device_id = 0;
        int driver_version = 0;
        int runtime_version = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

        // 所有版本复用同一 Host 输入和同一对 Device Buffer，保持实验负载一致。
        const std::vector<float> host_input =
            cuda_foundations::test::make_deterministic_input(element_count);
        float* device_input = nullptr;
        float* device_output = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), byte_count));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), byte_count));
        CUDA_CHECK(cudaMemcpy(
            device_input, host_input.data(), byte_count, cudaMemcpyHostToDevice));

        // 按固定顺序执行所选版本；正式 all 模式会得到可直接比较的 V0/V1/V2/V3 结果。
        std::vector<KernelResult> results;
        for (const KernelSpec& spec : selected_kernels(options.kernel_name)) {
            results.push_back(benchmark_kernel(
                spec,
                options,
                host_input,
                device_input,
                device_output,
                byte_count));
        }

        // 所有 Kernel 与验证完成后再统一释放 Device Buffer。
        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_input));

        compute_relative_metrics(&results);
        const dim3 grid = cuda_foundations::transpose::transpose_grid_dimensions(
            options.width, options.height);
        const dim3 block = cuda_foundations::transpose::transpose_block_dimensions();

        // 先打印全部结果，再按同一顺序写 CSV，方便终端与原始记录逐行对应。
        for (const KernelResult& result : results) {
            print_result(result, options, grid, block);
            if (!options.csv_path.empty()) {
                append_csv(
                    options.csv_path,
                    options,
                    result,
                    properties,
                    driver_version,
                    runtime_version,
                    grid,
                    block);
            }
        }

        if (!options.csv_path.empty()) {
            std::cout << "CSV 已追加: " << options.csv_path << std::endl;
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "Benchmark 失败: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}
