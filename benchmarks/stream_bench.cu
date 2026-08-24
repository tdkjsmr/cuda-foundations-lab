#include "benchmark.h"
#include "cuda_check.cuh"
#include "stream_pipeline.cuh"
#include "test_utils.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
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
#include <utility>
#include <vector>

#ifndef CUDA_FOUNDATIONS_GIT_COMMIT
#define CUDA_FOUNDATIONS_GIT_COMMIT "unknown"
#endif

namespace {

// V3.0/V3.1 共用同一 CLI；V3.2 会继续扩展 async 和多 Stream。
struct Options {
    std::string mode = "pinned_sync";
    cuda_foundations::streams::StreamWorkload workload{};
    std::size_t stream_count = 1U;
    int warmup_count = 20;
    int iteration_count = 100;
    int group_count = 5;
    bool profile_mode = false;
    std::string csv_path;
};

// 分别记录 CPU 提交、GPU 时间线跨度、端到端完成时间和两种明确字节口径。
struct BenchmarkResult {
    cuda_foundations::benchmark::Summary cpu_submit_us;
    cuda_foundations::benchmark::Summary gpu_span_us;
    cuda_foundations::benchmark::Summary end_to_end_us;
    double payload_throughput_gbps = 0.0;
    double accounted_traffic_rate_gbps = 0.0;
    bool bitwise_correct = false;
};

// 用两个 CUDA Event 测量默认 Stream 上从首个任务到末个任务的 GPU 时间线跨度。
class GpuSpanTimer {
public:
    GpuSpanTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    GpuSpanTimer(const GpuSpanTimer&) = delete;
    GpuSpanTimer& operator=(const GpuSpanTimer&) = delete;

    ~GpuSpanTimer() {
        CUDA_CHECK(cudaEventDestroy(stop_));
        CUDA_CHECK(cudaEventDestroy(start_));
    }

    void record_start() {
        CUDA_CHECK(cudaEventRecord(start_, nullptr));
    }

    void record_stop() {
        CUDA_CHECK(cudaEventRecord(stop_, nullptr));
    }

    double elapsed_us_after_synchronize() const {
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return static_cast<double>(elapsed_ms) * 1000.0;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

// 将 WIDTHxHEIGHT 严格解析为两个非零 size_t。
std::pair<std::size_t, std::size_t> parse_shape(const std::string& text) {
    const std::size_t separator = text.find('x');
    if (separator == std::string::npos || separator == 0U ||
        separator + 1U >= text.size()) {
        throw std::invalid_argument("--shape 必须使用 WIDTHxHEIGHT 格式");
    }

    const std::string width_text = text.substr(0U, separator);
    const std::string height_text = text.substr(separator + 1U);
    std::size_t width_characters = 0U;
    std::size_t height_characters = 0U;
    const unsigned long long parsed_width = std::stoull(width_text, &width_characters);
    const unsigned long long parsed_height = std::stoull(height_text, &height_characters);

    if (width_characters != width_text.size() ||
        height_characters != height_text.size() || parsed_width == 0ULL ||
        parsed_height == 0ULL ||
        parsed_width > std::numeric_limits<std::size_t>::max() ||
        parsed_height > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("--shape 必须包含两个 size_t 范围内的正整数");
    }

    return {static_cast<std::size_t>(parsed_width),
            static_cast<std::size_t>(parsed_height)};
}

// 将 CLI 文本解析为非零 size_t，拒绝负号和尾随字符。
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

// 将预热、迭代和组数解析为 int 范围内的正整数。
int parse_positive_int(const std::string& text, const char* option_name) {
    std::size_t parsed_characters = 0U;
    const long long parsed = std::stoll(text, &parsed_characters);
    if (parsed_characters != text.size() || parsed <= 0LL ||
        parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(option_name) + " 必须是 int 范围内的正整数");
    }
    return static_cast<int>(parsed);
}

// 解析模式、Shape、Chunk、Stream、稳定态统计、Profile 和 CSV 参数。
Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--mode" && index + 1 < argc) {
            options.mode = argv[++index];
        } else if (argument == "--shape" && index + 1 < argc) {
            const auto [width, height] = parse_shape(argv[++index]);
            options.workload.width = width;
            options.workload.height = height;
        } else if (argument == "--chunks" && index + 1 < argc) {
            options.workload.chunk_count = parse_positive_size(argv[++index], "--chunks");
        } else if (argument == "--streams" && index + 1 < argc) {
            options.stream_count = parse_positive_size(argv[++index], "--streams");
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

    // V3.1 保留 Pageable 回归并新增 Pinned Sync，二者都必须保持单 Stream。
    if (options.mode != "pageable_sync" && options.mode != "pinned_sync") {
        throw std::invalid_argument(
            "V3.1 --mode 只支持 pageable_sync 或 pinned_sync");
    }
    if (options.stream_count != 1U) {
        throw std::invalid_argument("同步 Stream 基线的 --streams 必须为 1");
    }

    // Profile 保留用户指定的预热数，但只捕获一次正式 Pipeline。
    if (options.profile_mode) {
        options.iteration_count = 1;
        options.group_count = 1;
    }
    return options;
}

// 生成 UTC ISO-8601 时间戳，使不同机器的 CSV 可直接对齐。
std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc_time{};
    gmtime_r(&now, &utc_time);

    std::ostringstream stream;
    stream << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

// CUDA Runtime 用 12040 表示 12.4，此函数转换为易读文本。
std::string format_cuda_version(int version) {
    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

// GPU 名称等字段包含逗号或双引号时执行标准 CSV 转义。
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

// 直接根据 Input 索引检查 Batch Output，避免为 512 MiB 工作量再分配一份 Reference。
bool validate_batch_transpose(
    const float* input,
    const float* output,
    std::size_t element_count,
    const cuda_foundations::streams::StreamWorkload& workload,
    const cuda_foundations::streams::WorkloadLayout& layout,
    std::string* error_message) {
    if (input == nullptr || output == nullptr ||
        element_count != layout.total_elements) {
        if (error_message != nullptr) {
            *error_message = "Stream Benchmark Host Buffer 容量与 Layout 不一致";
        }
        return false;
    }

    for (std::size_t chunk = 0U; chunk < workload.chunk_count; ++chunk) {
        const std::size_t chunk_offset = chunk * layout.elements_per_chunk;
        for (std::size_t y = 0U; y < workload.height; ++y) {
            for (std::size_t x = 0U; x < workload.width; ++x) {
                const std::size_t input_index =
                    chunk_offset + y * workload.width + x;
                const std::size_t output_index =
                    chunk_offset + x * workload.height + y;
                if (cuda_foundations::test::float_bits(input[input_index]) !=
                    cuda_foundations::test::float_bits(output[output_index])) {
                    if (error_message != nullptr) {
                        *error_message = "Chunk " + std::to_string(chunk) +
                                         " 的输出元素 " +
                                         std::to_string(output_index - chunk_offset) +
                                         " 位模式不一致";
                    }
                    return false;
                }
            }
        }
    }
    return true;
}

// 对完整 H2D + Kernel + D2H Pipeline 执行预热、稳定态分组计时和事后正确性验证。
BenchmarkResult benchmark_pipeline(
    const Options& options,
    const cuda_foundations::streams::WorkloadLayout& layout,
    const float* host_input,
    float* host_output,
    const cuda_foundations::streams::DeviceBufferPair& buffers) {
    // 预热以一次完整 Batch Pipeline 为单位，不只预热 Kernel。
    for (int iteration = 0; iteration < options.warmup_count; ++iteration) {
        cuda_foundations::streams::execute_synchronous_pipeline(
            options.workload,
            layout,
            host_input,
            host_output,
            buffers,
            false);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuSpanTimer gpu_span_timer;
    std::vector<double> cpu_submit_group_us;
    std::vector<double> gpu_span_group_us;
    std::vector<double> end_to_end_group_us;
    cpu_submit_group_us.reserve(static_cast<std::size_t>(options.group_count));
    gpu_span_group_us.reserve(static_cast<std::size_t>(options.group_count));
    end_to_end_group_us.reserve(static_cast<std::size_t>(options.group_count));

    // 各组都从完全空闲的 Device 边界开始，并等到所有输出完成才停表。
    for (int group = 0; group < options.group_count; ++group) {
        CUDA_CHECK(cudaDeviceSynchronize());

        // NSYS 只在 --profile 的唯一正式组捕获名为 profile 的范围。
        if (options.profile_mode) {
            CUDA_CHECK(cudaProfilerStart());
            nvtxRangePushA("profile");
        }

        // GPU Event 在默认 Stream 中包住全部 H2D、Kernel 和 D2H，但不等价于引擎 Busy Time。
        gpu_span_timer.record_start();
        const auto end_to_end_start = std::chrono::steady_clock::now();
        const auto cpu_submit_start = end_to_end_start;
        for (int iteration = 0; iteration < options.iteration_count; ++iteration) {
            cuda_foundations::streams::execute_synchronous_pipeline(
                options.workload,
                layout,
                host_input,
                host_output,
                buffers,
                options.profile_mode);
        }
        const auto cpu_submit_stop = std::chrono::steady_clock::now();
        gpu_span_timer.record_stop();

        // 终点同步保证 steady_clock 覆盖真正完成的 H2D + Kernel + D2H。
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end_to_end_stop = std::chrono::steady_clock::now();

        if (options.profile_mode) {
            nvtxRangePop();
            CUDA_CHECK(cudaProfilerStop());
        }

        const double divisor = static_cast<double>(options.iteration_count);
        cpu_submit_group_us.push_back(
            std::chrono::duration<double, std::micro>(
                cpu_submit_stop - cpu_submit_start).count() /
            divisor);
        gpu_span_group_us.push_back(
            gpu_span_timer.elapsed_us_after_synchronize() / divisor);
        end_to_end_group_us.push_back(
            std::chrono::duration<double, std::micro>(
                end_to_end_stop - end_to_end_start).count() /
            divisor);
    }

    // 在所有异步工作完成后才读取 Host Output 并与 CPU 索引关系对比。
    std::string error_message;
    const bool bitwise_correct = validate_batch_transpose(
        host_input,
        host_output,
        layout.total_elements,
        options.workload,
        layout,
        &error_message);
    if (!bitwise_correct) {
        throw std::runtime_error("Stream Benchmark 正确性失败: " + error_message);
    }

    const cuda_foundations::benchmark::Summary cpu_submit_summary =
        cuda_foundations::benchmark::summarize(cpu_submit_group_us);
    const cuda_foundations::benchmark::Summary gpu_span_summary =
        cuda_foundations::benchmark::summarize(gpu_span_group_us);
    const cuda_foundations::benchmark::Summary end_to_end_summary =
        cuda_foundations::benchmark::summarize(end_to_end_group_us);
    const double median_seconds = end_to_end_summary.median * 1.0e-6;

    // Payload Throughput 只计一份原始输入，表示应用每秒完成的有效数据。
    const double payload_throughput_gbps =
        static_cast<double>(layout.total_bytes) / median_seconds / 1.0e9;

    // 4B 是 H2D + Kernel Read + Kernel Write + D2H 记账流量，跨越 PCIe 和 DRAM，不是单一硬件带宽。
    const double accounted_traffic_rate_gbps = payload_throughput_gbps * 4.0;

    return BenchmarkResult{cpu_submit_summary,
                           gpu_span_summary,
                           end_to_end_summary,
                           payload_throughput_gbps,
                           accounted_traffic_rate_gbps,
                           true};
}

// 将全部实验条件和结果追加到 CSV，由外层脚本保证原子发布。
void append_csv(
    const std::string& path,
    const Options& options,
    const cuda_foundations::streams::WorkloadLayout& layout,
    const cuda_foundations::streams::DeviceCapabilities& capabilities,
    const BenchmarkResult& result,
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
        throw std::runtime_error("无法打开 Stream CSV: " + path);
    }

    if (write_header) {
        output << "git_commit,timestamp,gpu_name,cuda_driver_api_version,"
                  "cuda_runtime_version,pipeline_name,host_memory_type,copy_api,"
                  "kernel_name,width,height,dtype,grid_x,grid_y,block_x,block_y,"
                  "stream_count,chunk_count,chunk_bytes,chunk_mib,total_payload_bytes,"
                  "accounted_traffic_bytes,device_overlap,async_engine_count,"
                  "concurrent_kernels,warmup_count,iteration_count,group_count,"
                  "cpu_submit_min_us,cpu_submit_median_us,cpu_submit_p95_us,"
                  "cpu_submit_stddev_us,gpu_span_min_us,gpu_span_median_us,"
                  "gpu_span_p95_us,gpu_span_stddev_us,end_to_end_min_us,"
                  "end_to_end_median_us,end_to_end_p95_us,end_to_end_stddev_us,"
                  "payload_throughput_gbps,accounted_traffic_rate_gbps,"
                  "speedup_vs_pinned_sync,bitwise_correct\n";
    }

    const std::size_t grid_x =
        1U + (options.workload.width - 1U) / cuda_foundations::streams::kTileDim;
    const std::size_t grid_y =
        1U + (options.workload.height - 1U) / cuda_foundations::streams::kTileDim;
    const double chunk_mib =
        static_cast<double>(layout.bytes_per_chunk) / (1024.0 * 1024.0);

    // CSV 中的记账流量是整数 bytes；执行 4B 乘法前单独检查 size_t 溢出。
    if (layout.total_bytes > std::numeric_limits<std::size_t>::max() / 4U) {
        throw std::overflow_error("Stream 记账流量字节数发生 size_t 溢出");
    }
    const std::size_t accounted_traffic_bytes = layout.total_bytes * 4U;

    // Pinned Sync 是题目规定的相对基线，因此其自身加速比固定为 1.0。
    const bool use_pinned_memory = options.mode == "pinned_sync";
    const char* pipeline_name =
        use_pinned_memory ? "pinned_sync_v1" : "pageable_sync_v0";
    const char* host_memory_type = use_pinned_memory ? "Pinned" : "Pageable";
    const char* speedup_vs_pinned_sync = use_pinned_memory ? "1" : "nan";

    // 保留 double 结果的充足有效数字，便于从原始 CSV 重新计算统计量。
    output << std::setprecision(17);

    output << CUDA_FOUNDATIONS_GIT_COMMIT << ',' << utc_timestamp() << ','
           << csv_escape(capabilities.device_name) << ','
           << format_cuda_version(driver_version) << ','
           << format_cuda_version(runtime_version) << ',' << pipeline_name << ','
           << host_memory_type << ",cudaMemcpy,stream_padded_transpose_kernel,"
           << options.workload.width << ',' << options.workload.height
           << ",FP32," << grid_x << ',' << grid_y << ','
           << cuda_foundations::streams::kTileDim << ','
           << cuda_foundations::streams::kBlockRows << ',' << options.stream_count
           << ',' << options.workload.chunk_count << ',' << layout.bytes_per_chunk
           << ',' << chunk_mib << ',' << layout.total_bytes << ','
           << accounted_traffic_bytes << ','
           << capabilities.device_overlap << ',' << capabilities.async_engine_count
           << ',' << capabilities.concurrent_kernels << ',' << options.warmup_count
           << ',' << options.iteration_count << ',' << options.group_count << ','
           << result.cpu_submit_us.minimum << ',' << result.cpu_submit_us.median << ','
           << result.cpu_submit_us.p95 << ','
           << result.cpu_submit_us.standard_deviation << ','
           << result.gpu_span_us.minimum << ',' << result.gpu_span_us.median << ','
           << result.gpu_span_us.p95 << ','
           << result.gpu_span_us.standard_deviation << ','
           << result.end_to_end_us.minimum << ',' << result.end_to_end_us.median << ','
           << result.end_to_end_us.p95 << ','
           << result.end_to_end_us.standard_deviation << ','
           << result.payload_throughput_gbps << ','
           << result.accounted_traffic_rate_gbps << ',' << speedup_vs_pinned_sync
           << ',' << (result.bitwise_correct ? "true" : "false") << '\n';

    if (!output) {
        throw std::runtime_error("Stream CSV 写入失败: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // 在任何大容量分配前解析 CLI 并完成布局溢出验证。
        const Options options = parse_options(argc, argv);
        const cuda_foundations::streams::WorkloadLayout layout =
            cuda_foundations::streams::validate_and_derive_layout(options.workload);
        const cuda_foundations::streams::DeviceCapabilities capabilities =
            cuda_foundations::streams::query_device_capabilities();

        int driver_version = 0;
        int runtime_version = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

        // 两种同步模式只改变 Host Allocation；原始指针统一进入同一计时函数。
        const bool use_pinned_memory = options.mode == "pinned_sync";
        std::vector<float> pageable_input;
        std::vector<float> pageable_output;
        cuda_foundations::streams::PinnedHostBufferPair pinned_buffers{};
        float* host_input = nullptr;
        float* host_output = nullptr;

        if (use_pinned_memory) {
            // 两块 Pinned Buffer 各覆盖完整 512 MiB Payload，分配不计入时间。
            pinned_buffers = cuda_foundations::streams::allocate_pinned_host_buffers(
                layout.total_bytes);
            host_input = pinned_buffers.input;
            host_output = pinned_buffers.output;
        } else {
            // Pageable 回归继续使用 std::vector，便于在同一二进制中复测 V3.0。
            pageable_input.assign(layout.total_elements, 0.0F);
            pageable_output.assign(layout.total_elements, 0.0F);
            host_input = pageable_input.data();
            host_output = pageable_output.data();
        }
        const float host_sentinel =
            cuda_foundations::test::float_from_bits(0xA5A5A5A5U);
        std::fill_n(host_output, layout.total_elements, host_sentinel);

        // 计时前写遍所有 Host Page，同时用 Chunk ID 使各矩阵数据不同。
        for (std::size_t chunk = 0U; chunk < options.workload.chunk_count; ++chunk) {
            const std::size_t chunk_offset = chunk * layout.elements_per_chunk;
            for (std::size_t element = 0U; element < layout.elements_per_chunk; ++element) {
                const std::size_t pattern = (element * 37U + chunk * 101U) % 509U;
                host_input[chunk_offset + element] =
                    static_cast<float>(static_cast<int>(pattern) - 254) * 0.0625F;
            }
        }

        // V3.0/V3.1 都只分配一对单 Chunk Device Buffer，分配不计时。
        cuda_foundations::streams::DeviceBufferPair buffers =
            cuda_foundations::streams::allocate_device_buffers(layout.bytes_per_chunk);

        const BenchmarkResult result = benchmark_pipeline(
            options, layout, host_input, host_output, buffers);

        // 正常路径按 Device 后 Host 的逆序释放；RAII 仍为异常路径兜底。
        cuda_foundations::streams::release_device_buffers(&buffers);
        if (use_pinned_memory) {
            cuda_foundations::streams::release_pinned_host_buffers(&pinned_buffers);
        }

        if (!options.csv_path.empty()) {
            append_csv(options.csv_path,
                       options,
                       layout,
                       capabilities,
                       result,
                       driver_version,
                       runtime_version);
        }

        const double chunk_mib =
            static_cast<double>(layout.bytes_per_chunk) / (1024.0 * 1024.0);
        const double total_payload_mib =
            static_cast<double>(layout.total_bytes) / (1024.0 * 1024.0);
        const double host_allocation_mib = total_payload_mib * 2.0;
        const double device_allocation_mib = chunk_mib * 2.0;

        const char* pipeline_name =
            use_pinned_memory ? "pinned_sync_v1" : "pageable_sync_v0";
        const char* host_memory_name = use_pinned_memory
                                           ? "Pinned (cudaMallocHost)"
                                           : "Pageable (std::vector)";
        std::cout << std::fixed << std::setprecision(3)
                  << "Pipeline: " << pipeline_name << '\n'
                  << "Host memory: " << host_memory_name << '\n'
                  << "Copy API: cudaMemcpy (blocking)\n"
                  << "Kernel: stream_padded_transpose_kernel\n"
                  << "Shape/chunk: " << options.workload.width << 'x'
                  << options.workload.height << " FP32\n"
                  << "Grid/chunk: ("
                  << 1U + (options.workload.width - 1U) /
                                 cuda_foundations::streams::kTileDim
                  << ", "
                  << 1U + (options.workload.height - 1U) /
                                 cuda_foundations::streams::kTileDim
                  << ", 1)\n"
                  << "Block: (" << cuda_foundations::streams::kTileDim << ", "
                  << cuda_foundations::streams::kBlockRows << ", 1)\n"
                  << "Streams: " << options.stream_count << '\n'
                  << "Chunks: " << options.workload.chunk_count << '\n'
                  << "Chunk: " << chunk_mib << " MiB\n"
                  << "Total input payload: " << total_payload_mib << " MiB\n"
                  << (use_pinned_memory ? "Pinned" : "Pageable")
                  << " Host allocation (input + output): " << host_allocation_mib
                  << " MiB\n"
                  << "Device allocation (input + output): "
                  << device_allocation_mib << " MiB\n"
                  << "Warm-up pipelines: " << options.warmup_count << '\n'
                  << "Iterations/group: " << options.iteration_count << '\n'
                  << "Groups: " << options.group_count << '\n'
                  << "CPU submit Min: " << result.cpu_submit_us.minimum << " us\n"
                  << "CPU submit Median (P50): " << result.cpu_submit_us.median
                  << " us\n"
                  << "CPU submit P95: " << result.cpu_submit_us.p95 << " us\n"
                  << "CPU submit Stddev: "
                  << result.cpu_submit_us.standard_deviation << " us\n"
                  << "GPU span Min: " << result.gpu_span_us.minimum << " us\n"
                  << "GPU span Median (P50): " << result.gpu_span_us.median
                  << " us\n"
                  << "GPU span P95: " << result.gpu_span_us.p95 << " us\n"
                  << "GPU span Stddev: "
                  << result.gpu_span_us.standard_deviation << " us\n"
                  << "End-to-end Min: " << result.end_to_end_us.minimum << " us\n"
                  << "End-to-end Median (P50): " << result.end_to_end_us.median
                  << " us\n"
                  << "End-to-end P95: " << result.end_to_end_us.p95 << " us\n"
                  << "End-to-end Stddev: "
                  << result.end_to_end_us.standard_deviation << " us\n"
                  << "Payload throughput: " << result.payload_throughput_gbps
                  << " GB/s\n"
                  << "Accounted traffic rate (4B/T, cross-fabric): "
                  << result.accounted_traffic_rate_gbps << " GB/s\n"
                  << "GPU: " << capabilities.device_name << '\n'
                  << "deviceOverlap: " << capabilities.device_overlap << '\n'
                  << "asyncEngineCount: " << capabilities.async_engine_count << '\n'
                  << "concurrentKernels: " << capabilities.concurrent_kernels << '\n'
                  << "Bitwise correctness: PASS" << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "Stream Benchmark 失败: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}
