#include "benchmark.h"
#include "cuda_check.cuh"
#include "cuda_timer.cuh"
#include "reduction.cuh"
#include "reduction_utils.h"
#include "test_utils.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDA_FOUNDATIONS_GIT_COMMIT
#define CUDA_FOUNDATIONS_GIT_COMMIT "unknown"
#endif

namespace {

// 保存全部 CLI 参数；日志和 CSV 可以据此完整复现实验。
struct Options {
    cuda_foundations::reduction::KernelVersion kernel_version =
        cuda_foundations::reduction::KernelVersion::kFirstAdd;
    std::size_t input_count = 16777219U;
    int warmup_count = 20;
    int iteration_count = 100;
    int group_count = 5;
    bool profile_mode = false;
    std::string csv_path;
};

// 保存一次正式实验的计时、结构、带宽和误差结果。
struct BenchmarkResult {
    cuda_foundations::benchmark::Summary timing_us;
    double lower_bound_bandwidth_gbps = 0.0;
    std::size_t first_stage_partial_count = 0U;
    std::size_t launch_count = 0U;
    cuda_foundations::reduction::ErrorMetrics errors;
    float gpu_result = 0.0F;
};

// 完整解析非零 size_t，拒绝负号、零和尾随字符。
std::size_t parse_size(const std::string& text, const std::string& option_name) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(option_name + " 必须是 size_t 范围内的正整数");
    }
    std::size_t parsed_count = 0U;
    const unsigned long long parsed = std::stoull(text, &parsed_count);
    if (parsed_count != text.size() || parsed == 0ULL ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(option_name + " 必须是 size_t 范围内的正整数");
    }
    return static_cast<std::size_t>(parsed);
}

// 把文本解析为 int 范围内的正整数。
int parse_positive_int(const std::string& text, const std::string& option_name) {
    std::size_t parsed_count = 0U;
    const int value = std::stoi(text, &parsed_count);
    if (parsed_count != text.size() || value <= 0) {
        throw std::invalid_argument(option_name + " 必须是正整数");
    }
    return value;
}

// 解析 Kernel、N、预热、迭代、组数、Profile 和 CSV 参数。
Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--kernel" && index + 1 < argc) {
            const std::string kernel = argv[++index];
            if (kernel == "interleaved") {
                options.kernel_version =
                    cuda_foundations::reduction::KernelVersion::kInterleaved;
            } else if (kernel == "sequential") {
                options.kernel_version =
                    cuda_foundations::reduction::KernelVersion::kSequential;
            } else if (kernel == "first_add") {
                options.kernel_version =
                    cuda_foundations::reduction::KernelVersion::kFirstAdd;
            } else {
                throw std::invalid_argument(
                    "--kernel 必须是 interleaved、sequential 或 first_add");
            }
        } else if (argument == "--size" && index + 1 < argc) {
            options.input_count = parse_size(argv[++index], "--size");
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

    // Profile 模式在预热后只执行一次完整多阶段归约，控制报告事件数量。
    if (options.profile_mode) {
        options.iteration_count = 1;
        options.group_count = 1;
    }
    return options;
}

// 生成 UTC ISO-8601 时间戳，避免服务器时区影响实验对齐。
std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc_time{};
    gmtime_r(&now, &utc_time);

    std::ostringstream stream;
    stream << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

// 把 CUDA 整数版本转换成 major.minor 文本。
std::string format_cuda_version(int version) {
    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

// GPU 名称包含逗号或双引号时执行 CSV 转义。
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

// 对完整 GPU 多阶段归约执行预热、Event 计时和计时后的数值验证。
BenchmarkResult benchmark_reduction(
    const Options& options,
    const std::vector<float>& host_input,
    const cuda_foundations::reduction::CpuReference& reference,
    float* device_input,
    float* workspace_a,
    float* workspace_b) {
    cuda_foundations::reduction::ReductionLaunchInfo latest_info;

    // 每次预热都是从原始输入开始的一次完整归约，阶段间不插入 Host 同步。
    for (int iteration = 0; iteration < options.warmup_count; ++iteration) {
        latest_info = cuda_foundations::reduction::reduce(options.kernel_version,
                                                             device_input,
                                                             workspace_a,
                                                             workspace_b,
                                                             options.input_count);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_foundations::CudaEventTimer timer;
    std::vector<double> group_average_us;
    group_average_us.reserve(static_cast<std::size_t>(options.group_count));

    // Event 包围每组全部完整归约；组内不在每次或每阶段后同步。
    for (int group = 0; group < options.group_count; ++group) {
        timer.start();
        for (int iteration = 0; iteration < options.iteration_count; ++iteration) {
            latest_info = cuda_foundations::reduction::reduce(options.kernel_version,
                                                                 device_input,
                                                                 workspace_a,
                                                                 workspace_b,
                                                                 options.input_count);
        }
        const float total_ms = timer.stop();
        const double average_us = static_cast<double>(total_ms) * 1000.0 /
                                  static_cast<double>(options.iteration_count);
        group_average_us.push_back(average_us);
    }

    // 只复制最后一次完整归约的单个 float，不把 D2H 计入 Kernel-only 时间。
    float gpu_result = 0.0F;
    CUDA_CHECK(cudaMemcpy(
        &gpu_result, latest_info.device_result, sizeof(float), cudaMemcpyDeviceToHost));

    const cuda_foundations::reduction::ErrorMetrics errors =
        cuda_foundations::reduction::error_metrics(gpu_result, reference);
    if (errors.absolute_error > errors.tolerance) {
        throw std::runtime_error("Reduction Benchmark 的数值误差超过阈值");
    }

    const cuda_foundations::benchmark::Summary timing =
        cuda_foundations::benchmark::summarize(group_average_us);

    // 下界带宽只计算原始输入读取字节，不包含 Partial Sum 中间读写。
    const double input_bytes =
        static_cast<double>(host_input.size()) * static_cast<double>(sizeof(float));
    const double bandwidth_gbps = input_bytes / (timing.median * 1.0e-6) / 1.0e9;

    return BenchmarkResult{
        timing,
        bandwidth_gbps,
        latest_info.first_stage_partial_count,
        latest_info.launch_count,
        errors,
        gpu_result};
}

// 把全部可复现实验信息和结果追加到 CSV。
void append_csv(const std::string& path,
                const Options& options,
                const BenchmarkResult& result,
                const cuda_foundations::reduction::CpuReference& reference,
                const cudaDeviceProp& properties,
                int driver_version,
                int runtime_version) {
    const std::filesystem::path csv_path(path);
    if (csv_path.has_parent_path()) {
        std::filesystem::create_directories(csv_path.parent_path());
    }

    const bool write_header = !std::filesystem::exists(csv_path) ||
                              std::filesystem::file_size(csv_path) == 0U;
    std::ofstream output(csv_path, std::ios::app);
    if (!output) {
        throw std::runtime_error("无法打开 CSV: " + path);
    }

    if (write_header) {
        output << "git_commit,timestamp,gpu_name,cuda_driver_api_version,cuda_runtime_version,"
                  "kernel_name,n,dtype,block,first_stage_grid,warmup_count,"
                  "iteration_count,group_count,min_us,median_us,p95_us,stddev_us,"
                  "lower_bound_bandwidth_gbps,first_stage_partial_count,"
                  "kernel_launch_count,static_shared_memory_bytes,gpu_result,"
                  "cpu_double_reference,max_abs_error,normalized_error,tolerance\n";
    }

    output << CUDA_FOUNDATIONS_GIT_COMMIT << ',' << utc_timestamp() << ','
           << csv_escape(properties.name) << ',' << format_cuda_version(driver_version)
           << ',' << format_cuda_version(runtime_version) << ','
           << cuda_foundations::reduction::kernel_name(options.kernel_version) << ','
           << options.input_count << ",FP32," << cuda_foundations::reduction::kBlockSize
           << ',' << result.first_stage_partial_count << ',' << options.warmup_count
           << ',' << options.iteration_count << ',' << options.group_count << ','
           << result.timing_us.minimum << ',' << result.timing_us.median << ','
           << result.timing_us.p95 << ',' << result.timing_us.standard_deviation << ','
           << result.lower_bound_bandwidth_gbps << ','
           << result.first_stage_partial_count << ',' << result.launch_count << ','
           << cuda_foundations::reduction::kBlockSize * sizeof(float) << ','
           << result.gpu_result << ',' << reference.sum << ','
           << result.errors.absolute_error << ','
           << result.errors.normalized_error << ',' << result.errors.tolerance << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        // 查询设备和版本信息，保证性能记录可追溯。
        int device_id = 0;
        int driver_version = 0;
        int runtime_version = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

        // 正式 Benchmark 固定使用 deterministic random 输入。
        const std::vector<float> host_input = cuda_foundations::reduction::make_input(
            options.input_count, cuda_foundations::reduction::InputPattern::kRandom);
        const cuda_foundations::reduction::CpuReference reference =
            cuda_foundations::reduction::cpu_reference(host_input);

        const std::size_t input_bytes =
            cuda_foundations::test::checked_float_byte_count(options.input_count);
        const std::size_t workspace_count =
            cuda_foundations::reduction::workspace_elements(
                options.kernel_version, options.input_count);
        const std::size_t workspace_bytes =
            cuda_foundations::test::checked_float_byte_count(workspace_count);

        float* device_input = nullptr;
        float* workspace_a = nullptr;
        float* workspace_b = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), input_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace_a), workspace_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace_b), workspace_bytes));
        CUDA_CHECK(cudaMemcpy(
            device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));

        const BenchmarkResult result = benchmark_reduction(
            options, host_input, reference, device_input, workspace_a, workspace_b);

        CUDA_CHECK(cudaFree(workspace_b));
        CUDA_CHECK(cudaFree(workspace_a));
        CUDA_CHECK(cudaFree(device_input));

        std::cout << std::fixed << std::setprecision(6)
                  << "Kernel: "
                  << cuda_foundations::reduction::kernel_name(options.kernel_version)
                  << "\nN: " << options.input_count
                  << "\nPattern: random\nBlock: "
                  << cuda_foundations::reduction::kBlockSize
                  << "\nFirst-stage partials: " << result.first_stage_partial_count
                  << "\nKernel launches/reduction: " << result.launch_count
                  << "\nWarm-up reductions: " << options.warmup_count
                  << "\nIterations/group: " << options.iteration_count
                  << "\nGroups: " << options.group_count
                  << "\nMin: " << result.timing_us.minimum << " us"
                  << "\nMedian (P50): " << result.timing_us.median << " us"
                  << "\nP95: " << result.timing_us.p95 << " us"
                  << "\nStddev: " << result.timing_us.standard_deviation << " us"
                  << "\nLower-bound effective bandwidth: "
                  << result.lower_bound_bandwidth_gbps << " GB/s"
                  << "\nAbsolute error: " << result.errors.absolute_error
                  << "\nNormalized error: " << result.errors.normalized_error
                  << "\nTolerance: " << result.errors.tolerance
                  << "\nCorrectness: PASS\n" << std::endl;

        if (!options.csv_path.empty()) {
            append_csv(
                options.csv_path,
                options,
                result,
                reference,
                properties,
                driver_version,
                runtime_version);
            std::cout << "CSV 已追加: " << options.csv_path << std::endl;
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "Benchmark 失败: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}
