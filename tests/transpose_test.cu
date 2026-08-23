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

// 两个 Host 启动函数拥有相同签名，测试可用统一流程调用 V0 或 V1。
using LaunchFunction = void (*)(const float*,
                                float*,
                                std::size_t,
                                std::size_t,
                                cudaStream_t);

// 先用手算的 3×2 小矩阵验证 CPU Reference 自身的输出布局。
bool verify_cpu_transpose_reference() {
    const std::vector<float> input = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::vector<float> expected = {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};
    const std::vector<float> actual =
        cuda_foundations::test::transpose_reference(input, 3U, 2U);

    std::string error_message;
    if (!cuda_foundations::test::bitwise_equal(actual, expected, &error_message)) {
        std::cerr << "[FAIL] cpu_reference_3x2: " << error_message << std::endl;
        return false;
    }

    std::cout << "[PASS] cpu_reference_3x2" << std::endl;
    return true;
}

// 为一个输入执行完整 H2D → 指定 Kernel → D2H 流程，并按位对照 CPU Reference。
bool run_kernel_case(std::size_t width,
                     std::size_t height,
                     const std::vector<float>& host_input,
                     const std::string& kernel_name,
                     LaunchFunction launch,
                     bool transpose_output,
                     const std::string& case_name) {
    // 先确认测试数据与 Shape 一致，避免测试自身掩盖实现错误。
    const std::size_t element_count =
        cuda_foundations::test::checked_element_count(width, height);
    if (host_input.size() != element_count) {
        std::cerr << "[FAIL] " << kernel_name << '/' << case_name
                  << ": 测试输入元素数量错误" << std::endl;
        return false;
    }

    // V0 Reference 原样复制；V1 Reference 把 input[y][x] 放到 output[x][y]。
    const std::vector<float> reference =
        transpose_output
            ? cuda_foundations::test::transpose_reference(host_input, width, height)
            : cuda_foundations::test::copy_reference(host_input);
    std::vector<float> host_output(element_count, 0.0F);
    const std::size_t byte_count =
        cuda_foundations::test::checked_float_byte_count(element_count);

    // 每个测试用例独立分配 Device Buffer，防止前一个 Kernel 的结果泄漏到下一用例。
    float* device_input = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), byte_count));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), byte_count));

    // 非零哨兵保证 Kernel 必须覆盖每一个合法输出位置。
    CUDA_CHECK(cudaMemset(device_output, 0xA5, byte_count));
    CUDA_CHECK(cudaMemcpy(
        device_input, host_input.data(), byte_count, cudaMemcpyHostToDevice));

    // 调用被测版本，并用同步捕获执行阶段的越界或非法地址错误。
    launch(device_input, device_output, width, height, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 同步完成后再把全部输出复制回 Host。
    CUDA_CHECK(cudaMemcpy(
        host_output.data(), device_output, byte_count, cudaMemcpyDeviceToHost));

    // 测试结束前按分配的逆序释放 Device 资源。
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));

    // Transpose 是纯数据重排，因此包括 NaN payload 在内都要求逐位一致。
    std::string error_message;
    if (!cuda_foundations::test::bitwise_equal(
            host_output, reference, &error_message)) {
        std::cerr << "[FAIL] " << kernel_name << '/' << case_name << ": "
                  << error_message << std::endl;
        return false;
    }

    std::cout << "[PASS] " << kernel_name << '/' << case_name << " (" << width << 'x'
              << height << ')' << std::endl;
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

    bool all_passed = verify_cpu_transpose_reference();

    // 每个 Shape 都同时回归 V0 Copy 和新增的 V1 Naive Transpose。
    for (const auto& [width, height] : shapes) {
        const std::size_t element_count =
            cuda_foundations::test::checked_element_count(width, height);
        const std::vector<float> input =
            cuda_foundations::test::make_deterministic_input(element_count);
        const std::string shape_name =
            "deterministic_" + std::to_string(width) + "x" + std::to_string(height);

        all_passed = run_kernel_case(
                         width,
                         height,
                         input,
                         "copy_v0",
                         cuda_foundations::transpose::launch_copy,
                         false,
                         shape_name) &&
                     all_passed;

        all_passed = run_kernel_case(
                         width,
                         height,
                         input,
                         "naive_v1",
                         cuda_foundations::transpose::launch_naive,
                         true,
                         shape_name) &&
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

    all_passed = run_kernel_case(
                     4U,
                     2U,
                     special_values,
                     "copy_v0",
                     cuda_foundations::transpose::launch_copy,
                     false,
                     "special_bit_patterns") &&
                 all_passed;

    all_passed = run_kernel_case(
                     4U,
                     2U,
                     special_values,
                     "naive_v1",
                     cuda_foundations::transpose::launch_naive,
                     true,
                     "special_bit_patterns") &&
                 all_passed;

    if (!all_passed) {
        std::cerr << "Transpose V0/V1 测试失败" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Transpose V0/V1 全部正确性与边界测试通过" << std::endl;
    return EXIT_SUCCESS;
}
