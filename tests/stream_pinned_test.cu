#include "cuda_check.cuh"
#include "stream_pipeline.cuh"
#include "test_utils.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// Pinned Host Buffer 拥有 cudaMallocHost 返回的资源，编译期禁止复制所有权。
static_assert(
    !std::is_copy_constructible_v<cuda_foundations::streams::PinnedHostBufferPair>);
static_assert(
    !std::is_copy_assignable_v<cuda_foundations::streams::PinnedHostBufferPair>);
static_assert(
    std::is_move_constructible_v<cuda_foundations::streams::PinnedHostBufferPair>);

// 每个用例描述一批独立矩阵；Shape 集合与 Pageable 基线保持一致。
struct TestCase {
    cuda_foundations::streams::StreamWorkload workload;
    std::string name;
};

// 默认输入把 Chunk ID 编入模式，避免错误地重复处理 Chunk 0 仍然通过。
std::vector<float> make_distinct_chunk_input(
    const cuda_foundations::streams::StreamWorkload& workload,
    const cuda_foundations::streams::WorkloadLayout& layout) {
    std::vector<float> input(layout.total_elements, 0.0F);

    for (std::size_t chunk = 0U; chunk < workload.chunk_count; ++chunk) {
        const std::size_t chunk_offset = chunk * layout.elements_per_chunk;
        for (std::size_t element = 0U; element < layout.elements_per_chunk; ++element) {
            // 使用与正式 Benchmark 相同的确定性数据模式。
            const std::size_t pattern = (element * 37U + chunk * 101U) % 509U;
            input[chunk_offset + element] =
                static_cast<float>(static_cast<int>(pattern) - 254) * 0.0625F;
        }
    }
    return input;
}

// CPU 对每个 Chunk 独立转置，输出 Chunk 偏移与 GPU Pipeline 完全一致。
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

// 期望指定 Host 参数检查抛出 C++ 异常。
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

// 同时验证 Runtime Pointer 属性、Move-only 状态和显式释放后的清空状态。
bool verify_pinned_allocation_contract() {
    constexpr std::size_t kElementCount = 16U;
    constexpr std::size_t kBytes = kElementCount * sizeof(float);

    cuda_foundations::streams::PinnedHostBufferPair original =
        cuda_foundations::streams::allocate_pinned_host_buffers(kBytes);

    cudaPointerAttributes input_attributes{};
    cudaPointerAttributes output_attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&input_attributes, original.input));
    CUDA_CHECK(cudaPointerGetAttributes(&output_attributes, original.output));

    if (original.input == nullptr || original.output == nullptr ||
        original.capacity_bytes != kBytes ||
        input_attributes.type != cudaMemoryTypeHost ||
        output_attributes.type != cudaMemoryTypeHost) {
        std::cerr << "[FAIL] pinned_allocation_contract: Pointer 属性或容量错误"
                  << std::endl;
        return false;
    }

    cuda_foundations::streams::PinnedHostBufferPair moved = std::move(original);
    if (original.input != nullptr || original.output != nullptr ||
        original.capacity_bytes != 0U || moved.input == nullptr ||
        moved.output == nullptr || moved.capacity_bytes != kBytes) {
        std::cerr << "[FAIL] pinned_allocation_contract: Move 后所有权状态错误"
                  << std::endl;
        return false;
    }

    cuda_foundations::streams::release_pinned_host_buffers(&moved);
    if (moved.input != nullptr || moved.output != nullptr ||
        moved.capacity_bytes != 0U) {
        std::cerr << "[FAIL] pinned_allocation_contract: Release 后状态未清空"
                  << std::endl;
        return false;
    }

    std::cout << "[PASS] pinned_allocation_contract" << std::endl;
    return true;
}

// 执行一次完整 Pinned + Synchronous Pipeline，并对 Batch 结果逐位验证。
bool run_pinned_pipeline_case(
    const TestCase& test_case,
    const std::vector<float>* supplied_input = nullptr) {
    const cuda_foundations::streams::WorkloadLayout layout =
        cuda_foundations::streams::validate_and_derive_layout(test_case.workload);

    // 特殊位模式由调用方传入，其余用例使用确定性的多 Chunk 模式。
    const std::vector<float> generated_input =
        supplied_input == nullptr
            ? make_distinct_chunk_input(test_case.workload, layout)
            : std::vector<float>{};
    const std::vector<float>& pageable_source =
        supplied_input == nullptr ? generated_input : *supplied_input;
    if (pageable_source.size() != layout.total_elements) {
        std::cerr << "[FAIL] " << test_case.name << ": 输入容量与 Layout 不一致"
                  << std::endl;
        return false;
    }

    // Reference 在 Host 上独立生成，不复用 GPU Output。
    const std::vector<float> reference =
        make_batch_reference(pageable_source, test_case.workload, layout);

    // cudaMallocHost 分配 Input/Output；分配和初始化都不属于 Pipeline 计时。
    cuda_foundations::streams::PinnedHostBufferPair host_buffers =
        cuda_foundations::streams::allocate_pinned_host_buffers(layout.total_bytes);
    std::copy(pageable_source.begin(), pageable_source.end(), host_buffers.input);
    const float host_sentinel = cuda_foundations::test::float_from_bits(0xA5A5A5A5U);
    std::fill_n(host_buffers.output, layout.total_elements, host_sentinel);

    // V3.1 与 V3.0 一样只复用一对单 Chunk Device Buffer。
    cuda_foundations::streams::DeviceBufferPair device_buffers =
        cuda_foundations::streams::allocate_device_buffers(layout.bytes_per_chunk);
    CUDA_CHECK(cudaMemset(device_buffers.output, 0xA5, layout.bytes_per_chunk));

    // 唯一主要变化是 Host Pointer 来自 cudaMallocHost；执行顺序仍然同步串行。
    cuda_foundations::streams::execute_synchronous_pipeline(
        test_case.workload,
        layout,
        host_buffers.input,
        host_buffers.output,
        device_buffers,
        false);

    // 测试路径显式同步，保证捕获 Kernel 异步错误后才读取 Host Output。
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> actual(
        host_buffers.output, host_buffers.output + layout.total_elements);

    // 正常路径按 Device 后 Host 的顺序释放；两个类型的析构仍会兜底。
    cuda_foundations::streams::release_device_buffers(&device_buffers);
    cuda_foundations::streams::release_pinned_host_buffers(&host_buffers);

    std::string error_message;
    if (!cuda_foundations::test::bitwise_equal(actual, reference, &error_message)) {
        std::cerr << "[FAIL] " << test_case.name << ": " << error_message << std::endl;
        return false;
    }

    std::cout << "[PASS] " << test_case.name << " (" << test_case.workload.width
              << 'x' << test_case.workload.height << ", chunks="
              << test_case.workload.chunk_count << ')' << std::endl;
    return true;
}

}  // namespace

int main() {
    bool all_passed = expect_exception("reject_zero_pinned_capacity", [] {
        cuda_foundations::streams::allocate_pinned_host_buffers(0U);
    });
    all_passed = verify_pinned_allocation_contract() && all_passed;

    // 与 Pageable 回归使用同一组 Shape，保证两种 Host Memory 的功能合同一致。
    const std::vector<TestCase> cases = {
        {{1U, 1U, 1U}, "pinned_single_element_single_chunk"},
        {{31U, 33U, 3U}, "pinned_smaller_than_tile_multi_chunk"},
        {{32U, 32U, 2U}, "pinned_exact_tile"},
        {{33U, 31U, 5U}, "pinned_non_divisible_rectangular"},
        {{257U, 129U, 9U}, "pinned_future_eight_stream_remainder"},
        {{4096U, 512U, 2U}, "pinned_eight_mib_chunk_smoke"},
    };
    for (const TestCase& test_case : cases) {
        all_passed = run_pinned_pipeline_case(test_case) && all_passed;
    }

    // 特殊位模式证明 Pinned Memory 不改变纯搬运的任何 IEEE-754 bit pattern。
    const TestCase special_case{{4U, 3U, 2U}, "pinned_special_ieee754_bit_patterns"};
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
    all_passed = run_pinned_pipeline_case(special_case, &special_input) && all_passed;

    if (!all_passed) {
        std::cerr << "Stream V3.1 Pinned Sync 测试失败" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Stream V3.1 Pinned Sync 全部正确性、边界、属性和所有权测试通过"
              << std::endl;
    return EXIT_SUCCESS;
}
