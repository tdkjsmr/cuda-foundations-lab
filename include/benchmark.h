#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cuda_foundations::benchmark {

// 保存多组稳定态测量的核心统计量，单位由调用者决定。
struct Summary {
    double minimum{};
    double median{};
    double p95{};
    double standard_deviation{};
};

// 使用线性插值分位数，避免样本数较少时只能返回某一个离散观测值。
inline double percentile(const std::vector<double>& sorted_values, double probability) {
    if (sorted_values.empty()) {
        throw std::invalid_argument("计算分位数时样本不能为空");
    }

    const double position = probability * static_cast<double>(sorted_values.size() - 1U);
    const auto lower_index = static_cast<std::size_t>(std::floor(position));
    const auto upper_index = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower_index);

    return sorted_values[lower_index] * (1.0 - fraction) +
           sorted_values[upper_index] * fraction;
}

// 对各组“单次 Kernel 平均耗时”计算最小值、P50、P95 和总体标准差。
inline Summary summarize(const std::vector<double>& samples) {
    if (samples.empty()) {
        throw std::invalid_argument("Benchmark 样本不能为空");
    }

    std::vector<double> sorted_values = samples;
    std::sort(sorted_values.begin(), sorted_values.end());

    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());

    double squared_error_sum = 0.0;
    for (const double sample : samples) {
        const double difference = sample - mean;
        squared_error_sum += difference * difference;
    }

    Summary summary;
    summary.minimum = sorted_values.front();
    summary.median = percentile(sorted_values, 0.50);
    summary.p95 = percentile(sorted_values, 0.95);
    summary.standard_deviation =
        std::sqrt(squared_error_sum / static_cast<double>(samples.size()));
    return summary;
}

}  // namespace cuda_foundations::benchmark
