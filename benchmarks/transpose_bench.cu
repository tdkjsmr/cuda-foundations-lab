#include "benchmark.h"
#include "cuda_check.cuh"
#include "cuda_timer.cuh"
#include "test_utils.h"
#include "transpose.cuh"

#include <cuda_runtime.h>

#include <chrono>
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

// 集中保存 CLI 参数，保证日志能够完整复现实验配置。
struct Options {
    std::string kernel_name = "copy";
    std::size_t width = 4096U;
    std::size_t height = 4096U;
    int warmup_count = 20;
    int iteration_count = 100;
    int group_count = 5;
    bool profile_mode = false;
    std::string csv_path;
};

// 解析 WIDTHxHEIGHT，并复用公共容量检查保证 Shape 合法。
std::pair<std::size_t, std::size_t> parse_shape(const std::string& text) {
    const std::size_t separator = text.find('x');
    if (separator == std::string::npos) {
        throw std::invalid_argument("--shape 必须使用 WIDTHxHEIGHT 格式");
    }

    const std::size_t width = std::stoull(text.substr(0, separator));
    const std::size_t height = std::stoull(text.substr(separator + 1U));
    cuda_foundations::test::checked_element_count(width, height);
    return {width, height};
}

// 将十进制文本转换为正整数，拒绝零、负数和尾随字符。
int parse_positive_int(const std::string& text, const std::string& option_name) {
    std::size_t parsed_count = 0;
    const int value = std::stoi(text, &parsed_count);
    if (parsed_count != text.size() || value <= 0) {
        throw std::invalid_argument(option_name + " 必须是正整数");
    }
    return value;
}

// 解析 Benchmark 所需的 Shape、预热、迭代、组数、Profile 和 CSV 参数。
Options parse_options(int argc, char** argv) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "--kernel" && index + 1 < argc) {
            options.kernel_name = argv[++index];
            if (options.kernel_name != "copy") {
                throw std::invalid_argument("v1.0 只支持 --kernel copy");
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

    // Profile 模式在预热后只留下一个正式 Launch，避免 NCU 重复采集同一 Kernel。
    if (options.profile_mode) {
        options.iteration_count = 1;
        options.group_count = 1;
    }

    return options;
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

// 将本次实验追加到 CSV；首次创建文件时先写固定字段表头。
void append_csv(const std::string& path,
                const Options& options,
                const cudaDeviceProp& properties,
                int driver_version,
                int runtime_version,
                const cuda_foundations::benchmark::Summary& summary,
                double bandwidth_gbps) {
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
                  "effective_bandwidth_gbps,max_abs_error,relative_error\n";
    }

    const dim3 grid = cuda_foundations::transpose::copy_grid_dimensions(
        options.width, options.height);
    const dim3 block = cuda_foundations::transpose::copy_block_dimensions();

    output << CUDA_FOUNDATIONS_GIT_COMMIT << ','
           << utc_timestamp() << ','
           << csv_escape(properties.name) << ','
           << format_cuda_version(driver_version) << ','
           << format_cuda_version(runtime_version) << ','
           << "copy_v0," << options.width << 'x' << options.height << ",FP32,"
           << csv_escape(std::to_string(grid.x) + "x" + std::to_string(grid.y)) << ','
           << csv_escape(std::to_string(block.x) + "x" + std::to_string(block.y)) << ','
           << options.warmup_count << ',' << options.iteration_count << ','
           << options.group_count << ',' << summary.minimum << ',' << summary.median << ','
           << summary.p95 << ',' << summary.standard_deviation << ',' << bandwidth_gbps
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

        // Host 数据和 Device 分配均在预热与正式计时前完成。
        const std::vector<float> host_input =
            cuda_foundations::test::make_deterministic_input(element_count);
        std::vector<float> host_output(element_count, 0.0F);
        float* device_input = nullptr;
        float* device_output = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), byte_count));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), byte_count));
        CUDA_CHECK(cudaMemcpy(
            device_input, host_input.data(), byte_count, cudaMemcpyHostToDevice));

        // 预热消除首次 Context 初始化、缓存和频率爬升对稳定态数据的影响。
        for (int iteration = 0; iteration < options.warmup_count; ++iteration) {
            cuda_foundations::transpose::launch_copy(
                device_input, device_output, options.width, options.height);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cuda_foundations::CudaEventTimer timer;
        std::vector<double> group_average_us;
        group_average_us.reserve(static_cast<std::size_t>(options.group_count));

        // 每组只在边界记录 Event，中间连续 Launch，不插入逐轮同步。
        for (int group = 0; group < options.group_count; ++group) {
            timer.start();
            for (int iteration = 0; iteration < options.iteration_count; ++iteration) {
                cuda_foundations::transpose::launch_copy(
                    device_input, device_output, options.width, options.height);
            }
            const float total_ms = timer.stop();
            const double average_us =
                static_cast<double>(total_ms) * 1000.0 /
                static_cast<double>(options.iteration_count);
            group_average_us.push_back(average_us);
        }

        // 正式计时完成后再复制一次结果并按位验证，避免错误 Kernel 产生“快速”假结果。
        CUDA_CHECK(cudaMemcpy(
            host_output.data(), device_output, byte_count, cudaMemcpyDeviceToHost));
        std::string error_message;
        if (!cuda_foundations::test::bitwise_equal(
                host_output, host_input, &error_message)) {
            std::cerr << "Benchmark 正确性验证失败: " << error_message << std::endl;
            CUDA_CHECK(cudaFree(device_output));
            CUDA_CHECK(cudaFree(device_input));
            return EXIT_FAILURE;
        }

        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_input));

        const cuda_foundations::benchmark::Summary summary =
            cuda_foundations::benchmark::summarize(group_average_us);

        // V0 每个元素读一次、写一次；GB/s 使用十进制 10^9 字节定义。
        const double transferred_bytes = 2.0 * static_cast<double>(byte_count);
        const double bandwidth_gbps = transferred_bytes / (summary.median * 1.0e-6) / 1.0e9;
        const dim3 grid = cuda_foundations::transpose::copy_grid_dimensions(
            options.width, options.height);
        const dim3 block = cuda_foundations::transpose::copy_block_dimensions();

        std::cout << std::fixed << std::setprecision(3)
                  << "Kernel: copy_v0\n"
                  << "Shape: " << options.width << 'x' << options.height << " FP32\n"
                  << "Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n"
                  << "Block: (" << block.x << ", " << block.y << ", " << block.z << ")\n"
                  << "Warm-up: " << options.warmup_count << "\n"
                  << "Iterations/group: " << options.iteration_count << "\n"
                  << "Groups: " << options.group_count << "\n"
                  << "Min: " << summary.minimum << " us\n"
                  << "Median (P50): " << summary.median << " us\n"
                  << "P95: " << summary.p95 << " us\n"
                  << "Stddev: " << summary.standard_deviation << " us\n"
                  << "Effective bandwidth: " << bandwidth_gbps << " GB/s\n"
                  << "Bitwise correctness: PASS" << std::endl;

        if (!options.csv_path.empty()) {
            append_csv(
                options.csv_path,
                options,
                properties,
                driver_version,
                runtime_version,
                summary,
                bandwidth_gbps);
            std::cout << "CSV 已追加: " << options.csv_path << std::endl;
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "Benchmark 失败: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}
