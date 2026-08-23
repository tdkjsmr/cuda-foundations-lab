#include "cuda_check.cuh"
#include "reduction.cuh"
#include "reduction_utils.h"
#include "test_utils.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

// 一个测试用例描述输入长度、数据分布和便于定位失败的名字。
struct TestCase {
    std::size_t input_count;
    cuda_foundations::reduction::InputPattern pattern;
    std::string name;
};

// 执行 H2D → GPU 多阶段归约 → D2H，并检查结构信息与数值误差。
bool run_case(const TestCase& test_case) {
    const std::vector<float> host_input = cuda_foundations::reduction::make_input(
        test_case.input_count, test_case.pattern);
    const cuda_foundations::reduction::CpuReference reference =
        cuda_foundations::reduction::cpu_reference(host_input);

    // 输入 Buffer 和两块 Workspace 分别计算安全字节数。
    const std::size_t input_bytes =
        cuda_foundations::test::checked_float_byte_count(test_case.input_count);
    const std::size_t workspace_count =
        cuda_foundations::reduction::workspace_elements(test_case.input_count);
    const std::size_t workspace_bytes =
        cuda_foundations::test::checked_float_byte_count(workspace_count);

    float* device_input = nullptr;
    float* workspace_a = nullptr;
    float* workspace_b = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), input_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace_a), workspace_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace_b), workspace_bytes));

    // 非零哨兵帮助暴露未完整覆盖 Partial Sum 的错误。
    CUDA_CHECK(cudaMemset(workspace_a, 0xA5, workspace_bytes));
    CUDA_CHECK(cudaMemset(workspace_b, 0x5A, workspace_bytes));
    CUDA_CHECK(cudaMemcpy(
        device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));

    const cuda_foundations::reduction::ReductionLaunchInfo launch_info =
        cuda_foundations::reduction::reduce_interleaved(
            device_input, workspace_a, workspace_b, test_case.input_count);

    // 测试路径同步全部阶段，尽早暴露越界、非法地址和异步 Launch 错误。
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_result = 0.0F;
    CUDA_CHECK(cudaMemcpy(
        &gpu_result, launch_info.device_result, sizeof(float), cudaMemcpyDeviceToHost));

    // 完成验证数据复制后按逆序释放全部 Device 资源。
    CUDA_CHECK(cudaFree(workspace_b));
    CUDA_CHECK(cudaFree(workspace_a));
    CUDA_CHECK(cudaFree(device_input));

    const std::size_t expected_partials =
        cuda_foundations::reduction::workspace_elements(test_case.input_count);
    const std::size_t expected_launches =
        cuda_foundations::reduction::interleaved_launch_count(test_case.input_count);
    if (launch_info.first_stage_partial_count != expected_partials ||
        launch_info.launch_count != expected_launches) {
        std::cerr << "[FAIL] " << test_case.name
                  << ": Partial Sum 或 Launch 数量不符合多阶段计划" << std::endl;
        return false;
    }

    const cuda_foundations::reduction::ErrorMetrics errors =
        cuda_foundations::reduction::error_metrics(gpu_result, reference);
    if (errors.absolute_error > errors.tolerance) {
        std::cerr << std::setprecision(12) << "[FAIL] " << test_case.name
                  << ": gpu=" << gpu_result << ", reference=" << reference.sum
                  << ", absolute_error=" << errors.absolute_error
                  << ", normalized_error=" << errors.normalized_error
                  << ", tolerance=" << errors.tolerance << std::endl;
        return false;
    }

    std::cout << std::setprecision(6) << "[PASS] " << test_case.name
              << " N=" << test_case.input_count
              << " pattern="
              << cuda_foundations::reduction::pattern_name(test_case.pattern)
              << " launches=" << launch_info.launch_count
              << " abs_error=" << errors.absolute_error
              << " norm_error=" << errors.normalized_error << std::endl;
    return true;
}

}  // namespace

int main() {
    using cuda_foundations::reduction::InputPattern;

    // 文档规定的 N 覆盖 Warp、Block、非 2 的幂、非整除和大数组。
    const std::vector<std::size_t> required_sizes = {
        1U,
        31U,
        32U,
        33U,
        255U,
        256U,
        257U,
        1023U,
        1024U,
        1025U,
        1000003U,
        16777219U,
    };

    std::vector<TestCase> test_cases;
    test_cases.reserve(required_sizes.size() + 5U);

    // 每个规定 N 都使用包含正负小数的 deterministic random 分布回归。
    for (const std::size_t input_count : required_sizes) {
        test_cases.push_back(TestCase{
            input_count,
            InputPattern::kRandom,
            "required_random_" + std::to_string(input_count)});
    }

    // 在非 2 的幂且跨多个 Block 的 N 上补齐全部特殊输入分布。
    constexpr std::size_t kPatternTestSize = 1025U;
    test_cases.push_back({kPatternTestSize, InputPattern::kZeros, "all_zeros"});
    test_cases.push_back({kPatternTestSize, InputPattern::kOnes, "all_ones"});
    test_cases.push_back(
        {kPatternTestSize, InputPattern::kAlternating, "alternating_signs"});
    test_cases.push_back(
        {kPatternTestSize, InputPattern::kDynamicRange, "dynamic_range"});

    bool all_passed = true;
    for (const TestCase& test_case : test_cases) {
        all_passed = run_case(test_case) && all_passed;
    }

    if (!all_passed) {
        std::cerr << "Reduction V0 正确性或边界测试失败" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Reduction V0 全部正确性、边界、误差和多阶段测试通过" << std::endl;
    return EXIT_SUCCESS;
}
