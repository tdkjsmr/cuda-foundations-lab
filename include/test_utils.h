#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_foundations::test {

// 在计算 width × height 前检查 size_t 乘法是否溢出。
inline std::size_t checked_element_count(std::size_t width, std::size_t height) {
    if (width == 0 || height == 0) {
        throw std::invalid_argument("矩阵宽度和高度必须大于 0");
    }

    if (width > std::numeric_limits<std::size_t>::max() / height) {
        throw std::overflow_error("矩阵元素数量发生 size_t 溢出");
    }

    return width * height;
}

// 在把 FP32 元素数量换算为字节数前检查第二次乘法是否溢出。
inline std::size_t checked_float_byte_count(std::size_t element_count) {
    if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("FP32 Buffer 字节数发生 size_t 溢出");
    }

    return element_count * sizeof(float);
}

// 生成包含正数、负数和小数的确定性输入，便于复现实验。
inline std::vector<float> make_deterministic_input(std::size_t element_count) {
    std::vector<float> values(element_count);

    for (std::size_t index = 0; index < element_count; ++index) {
        const int centered = static_cast<int>(index % 257U) - 128;
        values[index] = static_cast<float>(centered) * 0.25F;
    }

    return values;
}

// Copy Baseline 的 CPU Reference 是逐元素原样复制。
inline std::vector<float> copy_reference(const std::vector<float>& input) {
    return input;
}

// 在 CPU 上按行主序执行矩阵转置，输出 Shape 为 height × width。
inline std::vector<float> transpose_reference(const std::vector<float>& input,
                                              std::size_t width,
                                              std::size_t height) {
    const std::size_t element_count = checked_element_count(width, height);
    if (input.size() != element_count) {
        throw std::invalid_argument("Transpose Reference 的输入元素数量与 Shape 不一致");
    }

    std::vector<float> output(element_count);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            // 输入 input[y][x] 在转置后写到 output[x][y]。
            output[x * height + y] = input[y * width + x];
        }
    }

    return output;
}

// 用 memcpy 读取 float 的位模式，避免类型双关违反严格别名规则。
inline std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// 将指定 IEEE-754 位模式恢复为 float，用于构造带 payload 的 NaN 测试数据。
inline float float_from_bits(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Copy 不改变任何数值，因此包括 NaN 在内都必须逐位一致。
inline bool bitwise_equal(const std::vector<float>& actual,
                          const std::vector<float>& expected,
                          std::string* error_message = nullptr) {
    if (actual.size() != expected.size()) {
        if (error_message != nullptr) {
            *error_message = "实际输出与 Reference 的元素数量不同";
        }
        return false;
    }

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (float_bits(actual[index]) != float_bits(expected[index])) {
            if (error_message != nullptr) {
                *error_message = "第 " + std::to_string(index) + " 个元素位模式不一致";
            }
            return false;
        }
    }

    return true;
}

}  // namespace cuda_foundations::test
