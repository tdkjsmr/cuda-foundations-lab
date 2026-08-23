#include "cuda_check.cuh"
#include "test_utils.h"
#include "transpose.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

// 为一个输入执行完整 H2D → Copy Kernel → D2H 流程，并按位对照 CPU Reference。
bool run_copy_case(std::size_t width,
                   std::size_t height,
                   const std::vector<float>& host_input,
                   const std::string& case_name) {
    // 先确认测试数据与 Shape 一致，避免测试自身掩盖实现错误。
    const std::size_t element_count =
        cuda_foundations::test::checked_element_count(width, height);
    if (host_input.size() != element_count) {
        std::cerr << "[FAIL] " << case_name << ": 测试输入元素数量错误" << std::endl;
        return false;
    }

    // Copy 的 CPU Reference 是输入本身，结果必须保持完全相同的位模式。
    const std::vector<float> reference =
        cuda_foundations::test::copy_reference(host_input);
    std::vector<float> host_output(element_count, 0.0F);
    const std::size_t byte_count =
        cuda_foundations::test::checked_float_byte_count(element_count);

    // 每个测试用例独立管理 Device Buffer，防止用例之间残留状态相互影响。
    float* device_input = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), byte_count));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), byte_count));

    // 先写入非零哨兵，确保 Kernel 必须覆盖所有有效输出元素才能通过测试。
    CUDA_CHECK(cudaMemset(device_output, 0xA5, byte_count));
    CUDA_CHECK(cudaMemcpy(
        device_input, host_input.data(), byte_count, cudaMemcpyHostToDevice));

    // 启动被测 Copy Kernel，并同步捕获执行阶段可能发生的越界或非法地址错误。
    cuda_foundations::transpose::launch_copy(
        device_input, device_output, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 在同步完成后读取输出，避免 Host 在异步任务结束前访问不完整结果。
    CUDA_CHECK(cudaMemcpy(
        host_output.data(), device_output, byte_count, cudaMemcpyDeviceToHost));

    // 测试结束前释放全部 Device 资源。
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));

    std::string error_message;
    const bool correct = cuda_foundations::test::bitwise_equal(
        host_output, reference, &error_message);

    if (!correct) {
        std::cerr << "[FAIL] " << case_name << ": " << error_message << std::endl;
        return false;
    }

    std::cout << "[PASS] " << case_name << " (" << width << 'x' << height << ')'
              << std::endl;
    return true;
}

}  // namespace

int main() {
    // 文档指定的 Shape 覆盖小 Tile、完整 Tile、非整除、长条、宽条和大矩阵。
    const std::vector<std::pair<std::size_t, std::size_t>> shapes = {
        {1U, 1U},
        {31U, 33U},
        {32U, 32U},
        {33U, 31U},
        {1024U, 8192U},
        {8192U, 1024U},
        {4096U, 4096U},
        {4097U, 3073U},
    };

    bool all_passed = true;

    // 每个 Shape 使用确定性混合数据，检查最后一个不完整 Tile 的边界保护。
    for (const auto& [width, height] : shapes) {
        const std::size_t element_count =
            cuda_foundations::test::checked_element_count(width, height);
        const std::vector<float> input =
            cuda_foundations::test::make_deterministic_input(element_count);
        all_passed = run_copy_case(
                         width,
                         height,
                         input,
                         "deterministic_" + std::to_string(width) + "x" +
                             std::to_string(height)) &&
                     all_passed;
    }

    // 特殊值覆盖正负零、负数、重复值、NaN payload 以及正负无穷。
    const std::vector<float> special_values = {
        0.0F,
        -0.0F,
        -7.5F,
        3.25F,
        3.25F,
        cuda_foundations::test::float_from_bits(0x7FC01234U),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    all_passed = run_copy_case(4U, 2U, special_values, "special_bit_patterns") &&
                 all_passed;

    if (!all_passed) {
        std::cerr << "Transpose V0 测试失败" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Transpose V0 全部正确性与边界测试通过" << std::endl;
    return EXIT_SUCCESS;
}
