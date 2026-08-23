#include "cuda_check.cuh"
#include "stream_pipeline.cuh"
#include "test_utils.h"

#include <cuda_runtime.h>

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

// 每个测试用例都描述矩阵 Shape、Chunk 数和可读名称。
struct TestCase {
    cuda_foundations::streams::StreamWorkload workload;
    std::string name;
};

// 生成每个 Chunk 都不同的数据，避免重复处理 Chunk 0 的错误被掩盖。
std::vector<float> make_distinct_chunk_input(
    const cuda_foundations::streams::StreamWorkload& workload,
    const cuda_foundations::streams::WorkloadLayout& layout) {
    std::vector<float> input(layout.total_elements, 0.0F);

    for (std::size_t chunk = 0U; chunk < workload.chunk_count; ++chunk) {
        const std::size_t chunk_offset = chunk * layout.elements_per_chunk;
        for (std::size_t element = 0U; element < layout.elements_per_chunk; ++element) {
            // 509 是质数，Chunk 和元素索引的组合不会快速重复。
            const std::size_t pattern = (element * 37U + chunk * 101U) % 509U;
            input[chunk_offset + element] =
                static_cast<float>(static_cast<int>(pattern) - 254) * 0.0625F;
        }
    }
    return input;
}

// 在 CPU 上对每个 Chunk 独立转置，保持 Chunk 在 Batch 中的偏移不变。
std::vector<float> make_batch_reference(
    const std::vector<float>& input,
    const cuda_foundations::streams::StreamWorkload& workload,
    const cuda_foundations::streams::WorkloadLayout& layout) {
    std::vector<float> reference(layout.total_elements, 0.0F);

    for (std::size_t chunk = 0U; chunk < workload.chunk_count; ++chunk) {
        const std::size_t chunk_offset = chunk * layout.elements_per_chunk;
        for (std::size_t y = 0U; y < workload.height; ++y) {
            for (std::size_t x = 0U; x < workload.width; ++x) {
                reference[chunk_offset + x * workload.height + y] =
                    input[chunk_offset + y * workload.width + x];
            }
        }
    }
    return reference;
}

// 执行一次完整 Pageable Sync Pipeline，然后对整批输出逐位验证。
bool run_pipeline_case(const TestCase& test_case,
                       const std::vector<float>* supplied_input = nullptr) {
    const cuda_foundations::streams::WorkloadLayout layout =
        cuda_foundations::streams::validate_and_derive_layout(test_case.workload);

    // 特殊位模式用例可传入自定义数据，其余用例使用唯一 Chunk 模式。
    const std::vector<float> generated_input =
        supplied_input == nullptr
            ? make_distinct_chunk_input(test_case.workload, layout)
            : std::vector<float>{};
    const std::vector<float>& host_input =
        supplied_input == nullptr ? generated_input : *supplied_input;

    if (host_input.size() != layout.total_elements) {
        std::cerr << "[FAIL] " << test_case.name << ": 输入容量与 Workload 不一致"
                  << std::endl;
        return false;
    }

    // CPU Reference 和 Host Output 都覆盖全部 Chunk。
    const std::vector<float> reference =
        make_batch_reference(host_input, test_case.workload, layout);
    const float host_sentinel = cuda_foundations::test::float_from_bits(0xA5A5A5A5U);
    std::vector<float> host_output(layout.total_elements, host_sentinel);

    // V3.0 安全复用一对单 Chunk Device Buffer，因为 blocking D2H 之后才处理下一块。
    cuda_foundations::streams::DeviceBufferPair buffers =
        cuda_foundations::streams::allocate_device_buffers(layout.bytes_per_chunk);

    // Device Output 预填充非零哨兵，确保 Kernel 必须覆盖每个合法元素。
    CUDA_CHECK(cudaMemset(buffers.output, 0xA5, layout.bytes_per_chunk));

    cuda_foundations::streams::execute_synchronous_pipeline(
        test_case.workload,
        layout,
        host_input.data(),
        host_output.data(),
        buffers,
        false);

    // 所有 Chunk 执行后再同步和验证，不在 Chunk 循环中额外序列化。
    CUDA_CHECK(cudaDeviceSynchronize());
    cuda_foundations::streams::release_device_buffers(&buffers);

    std::string error_message;
    if (!cuda_foundations::test::bitwise_equal(
            host_output, reference, &error_message)) {
        std::cerr << "[FAIL] " << test_case.name << ": " << error_message << std::endl;
        return false;
    }

    std::cout << "[PASS] " << test_case.name << " (" << test_case.workload.width
              << 'x' << test_case.workload.height << ", chunks="
              << test_case.workload.chunk_count << ')' << std::endl;
    return true;
}

// 期望指定 Host 参数验证路径抛出异常。
template <typename Function>
bool expect_exception(const std::string& name, Function&& function) {
    try {
        function();
    } catch (const std::exception&) {
        std::cout << "[PASS] " << name << std::endl;
        return true;
    }

    std::cerr << "[FAIL] " << name << ": 未抛出预期异常" << std::endl;
    return false;
}

// 严格检查零尺寸、Grid 上限和三类容量乘法溢出。
bool verify_invalid_layouts() {
    bool passed = true;

    passed = expect_exception("reject_zero_width", [] {
                 cuda_foundations::streams::validate_and_derive_layout({0U, 1U, 1U});
             }) &&
             passed;
    passed = expect_exception("reject_zero_height", [] {
                 cuda_foundations::streams::validate_and_derive_layout({1U, 0U, 1U});
             }) &&
             passed;
    passed = expect_exception("reject_zero_chunks", [] {
                 cuda_foundations::streams::validate_and_derive_layout({1U, 1U, 0U});
             }) &&
             passed;
    passed = expect_exception("reject_grid_y_overflow", [] {
                 constexpr std::size_t too_many_rows = 65535U * 32U + 1U;
                 cuda_foundations::streams::validate_and_derive_layout(
                     {1U, too_many_rows, 1U});
             }) &&
             passed;
    passed = expect_exception("reject_element_count_overflow", [] {
                 cuda_foundations::streams::validate_and_derive_layout(
                     {std::numeric_limits<std::size_t>::max(), 2U, 1U});
             }) &&
             passed;
    passed = expect_exception("reject_byte_count_overflow", [] {
                 const std::size_t too_many_floats =
                     std::numeric_limits<std::size_t>::max() / sizeof(float) + 1U;
                 cuda_foundations::streams::validate_and_derive_layout(
                     {too_many_floats, 1U, 1U});
             }) &&
             passed;
    passed = expect_exception("reject_batch_count_overflow", [] {
                 cuda_foundations::streams::validate_and_derive_layout(
                     {2U, 1U, std::numeric_limits<std::size_t>::max()});
             }) &&
             passed;
    passed = expect_exception("reject_zero_device_capacity", [] {
                 cuda_foundations::streams::allocate_device_buffers(0U);
             }) &&
             passed;
    passed = expect_exception("reject_mismatched_layout", [] {
                 const cuda_foundations::streams::StreamWorkload workload{1U, 1U, 1U};
                 cuda_foundations::streams::WorkloadLayout layout =
                     cuda_foundations::streams::validate_and_derive_layout(workload);
                 layout.bytes_per_chunk += sizeof(float);
                 const cuda_foundations::streams::DeviceBufferPair empty_buffers{};
                 float host_input = 1.0F;
                 float host_output = 0.0F;
                 cuda_foundations::streams::execute_synchronous_pipeline(
                     workload,
                     layout,
                     &host_input,
                     &host_output,
                     empty_buffers,
                     false);
             }) &&
             passed;

    return passed;
}

// 验证 Device Buffer 所有权只能移动，源对象被清空且目标对象只释放一次。
bool verify_move_only_device_buffers() {
    cuda_foundations::streams::DeviceBufferPair original =
        cuda_foundations::streams::allocate_device_buffers(sizeof(float));
    cuda_foundations::streams::DeviceBufferPair moved = std::move(original);

    if (original.input != nullptr || original.output != nullptr ||
        original.capacity_bytes != 0U || moved.input == nullptr ||
        moved.output == nullptr || moved.capacity_bytes != sizeof(float)) {
        std::cerr << "[FAIL] move_only_device_buffers: Move 后所有权状态错误"
                  << std::endl;
        return false;
    }

    cuda_foundations::streams::release_device_buffers(&moved);
    std::cout << "[PASS] move_only_device_buffers" << std::endl;
    return true;
}

// 检查运行时能力字段可被安全读取，但不把非零当作并发已发生的证据。
bool verify_device_capabilities() {
    const cuda_foundations::streams::DeviceCapabilities capabilities =
        cuda_foundations::streams::query_device_capabilities();

    if (capabilities.device_id < 0 || capabilities.device_name.empty() ||
        capabilities.device_overlap < 0 || capabilities.async_engine_count < 0 ||
        capabilities.concurrent_kernels < 0) {
        std::cerr << "[FAIL] device_capabilities: 设备能力字段无效"
                  << std::endl;
        return false;
    }

    std::cout << "[PASS] device_capabilities: " << capabilities.device_name
              << ", deviceOverlap=" << capabilities.device_overlap
              << ", asyncEngineCount=" << capabilities.async_engine_count
              << ", concurrentKernels=" << capabilities.concurrent_kernels << std::endl;
    return true;
}

}  // namespace

int main() {
    // 边界集合覆盖小于 Tile、恰好 Tile、非整除矩形和未来 8-Stream 不整除 Batch。
    const std::vector<TestCase> cases = {
        {{1U, 1U, 1U}, "single_element_single_chunk"},
        {{31U, 33U, 3U}, "smaller_than_tile_multi_chunk"},
        {{32U, 32U, 2U}, "exact_tile"},
        {{33U, 31U, 5U}, "non_divisible_rectangular"},
        {{257U, 129U, 9U}, "future_eight_stream_remainder"},
        {{4096U, 512U, 2U}, "eight_mib_chunk_smoke"},
    };

    bool all_passed = verify_invalid_layouts();
    all_passed = verify_move_only_device_buffers() && all_passed;
    all_passed = verify_device_capabilities() && all_passed;

    for (const TestCase& test_case : cases) {
        all_passed = run_pipeline_case(test_case) && all_passed;
    }

    // 特殊值覆盖正负零、Subnormal、最大有限值、正负无穷和两种 NaN payload。
    const TestCase special_case{{4U, 3U, 2U}, "special_ieee754_bit_patterns"};
    const auto special_layout =
        cuda_foundations::streams::validate_and_derive_layout(special_case.workload);
    const std::vector<float> patterns = {
        0.0F,
        -0.0F,
        -7.5F,
        3.25F,
        cuda_foundations::test::float_from_bits(0x00000001U),
        cuda_foundations::test::float_from_bits(0x7F7FFFFFU),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        cuda_foundations::test::float_from_bits(0x7FC01234U),
        cuda_foundations::test::float_from_bits(0xFFC05678U),
    };
    std::vector<float> special_input(special_layout.total_elements, 0.0F);
    for (std::size_t chunk = 0U; chunk < special_case.workload.chunk_count; ++chunk) {
        const std::size_t chunk_offset = chunk * special_layout.elements_per_chunk;
        for (std::size_t element = 0U;
             element < special_layout.elements_per_chunk;
             ++element) {
            special_input[chunk_offset + element] =
                patterns[(element + chunk * 3U) % patterns.size()];
        }
    }
    all_passed = run_pipeline_case(special_case, &special_input) && all_passed;

    if (!all_passed) {
        std::cerr << "Stream V3.0 Pageable Sync 测试失败" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Stream V3.0 Pageable Sync 全部正确性、边界和容量测试通过"
              << std::endl;
    return EXIT_SUCCESS;
}
