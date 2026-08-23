#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_foundations::reduction {

// 测试和演示共享这些输入分布，避免只用全 1 数组掩盖索引错误。
enum class InputPattern {
    kZeros,
    kOnes,
    kAlternating,
    kRandom,
    kDynamicRange,
};

// CPU Reference 同时保存 double 和与绝对值和，供误差阈值使用。
struct CpuReference {
    double sum = 0.0;
    double sum_abs = 0.0;
};

// 同时报告绝对误差、归一化误差和当前输入对应的验收阈值。
struct ErrorMetrics {
    double absolute_error = 0.0;
    double normalized_error = 0.0;
    double tolerance = 0.0;
};

// 生成确定性输入；相同 N 和 Pattern 在每次实验中保持一致。
inline std::vector<float> make_input(std::size_t count, InputPattern pattern) {
    if (count == 0U) {
        throw std::invalid_argument("Reduction 输入元素数必须大于 0");
    }

    std::vector<float> values(count, 0.0F);
    std::uint32_t state = 0x12345678U;

    for (std::size_t index = 0U; index < count; ++index) {
        switch (pattern) {
            case InputPattern::kZeros:
                // vector 已经零初始化；显式赋值用于清楚说明该分布。
                values[index] = 0.0F;
                break;
            case InputPattern::kOnes:
                // 全 1 可检查元素总数，但不会作为唯一测试分布。
                values[index] = 1.0F;
                break;
            case InputPattern::kAlternating:
                // 正负交替使总和接近零，可检验 normalized error 定义。
                values[index] = index % 2U == 0U ? 1.0F : -1.0F;
                break;
            case InputPattern::kRandom: {
                // 线性同余序列提供不依赖标准库实现的确定性伪随机数。
                state = state * 1664525U + 1013904223U;
                const int centered = static_cast<int>((state >> 8U) % 20001U) - 10000;
                values[index] = static_cast<float>(centered) / 10000.0F;
                break;
            }
            case InputPattern::kDynamicRange: {
                // 大小数混合暴露 FP32 加法顺序对舍入的影响。
                constexpr float sequence[] = {
                    100000000.0F, 1.0F, -100000000.0F, -1.0F, 0.25F, -0.25F};
                values[index] = sequence[index % 6U];
                break;
            }
        }
    }

    return values;
}

// 使用 double 串行累加，并同时计算 sum(abs(input))。
inline CpuReference cpu_reference(const std::vector<float>& input) {
    CpuReference reference;
    for (const float value : input) {
        const double converted = static_cast<double>(value);
        reference.sum += converted;
        reference.sum_abs += std::abs(converted);
    }
    return reference;
}

// 按项目规定计算 absolute error、normalized error 和验收阈值。
inline ErrorMetrics error_metrics(float gpu_result, const CpuReference& reference) {
    const double absolute_error =
        std::abs(static_cast<double>(gpu_result) - reference.sum);
    const double denominator = std::max(reference.sum_abs, 1.0);
    const double normalized_error = absolute_error / denominator;
    const double tolerance = 5.0e-6 * reference.sum_abs + 1.0e-5;
    return ErrorMetrics{absolute_error, normalized_error, tolerance};
}

// 返回 Pattern 的稳定文本名，供日志和 CSV 使用。
inline const char* pattern_name(InputPattern pattern) {
    switch (pattern) {
        case InputPattern::kZeros:
            return "zeros";
        case InputPattern::kOnes:
            return "ones";
        case InputPattern::kAlternating:
            return "alternating";
        case InputPattern::kRandom:
            return "random";
        case InputPattern::kDynamicRange:
            return "dynamic_range";
    }
    return "unknown";
}

// 把 CLI 文本转换为 Pattern，未知值立即拒绝。
inline InputPattern parse_pattern(const std::string& text) {
    if (text == "zeros") {
        return InputPattern::kZeros;
    }
    if (text == "ones") {
        return InputPattern::kOnes;
    }
    if (text == "alternating") {
        return InputPattern::kAlternating;
    }
    if (text == "random") {
        return InputPattern::kRandom;
    }
    if (text == "dynamic_range") {
        return InputPattern::kDynamicRange;
    }
    throw std::invalid_argument(
        "--pattern 只支持 zeros、ones、alternating、random 或 dynamic_range");
}

}  // namespace cuda_foundations::reduction
