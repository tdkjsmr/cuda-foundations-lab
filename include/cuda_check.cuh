#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace cuda_foundations {

// 统一检查 CUDA Runtime API 的返回值，并保留表达式、文件和行号，便于快速定位错误。
inline void check_cuda(cudaError_t result,
                       const char* expression,
                       const char* file,
                       int line) {
    // cudaSuccess 表示调用成功，此时直接返回，不产生额外输出。
    if (result == cudaSuccess) {
        return;
    }

    // cudaGetErrorName 给出稳定的错误枚举名，cudaGetErrorString 给出便于阅读的说明。
    std::cerr << "CUDA 调用失败: " << expression << '\n'
              << "  错误名称: " << cudaGetErrorName(result) << '\n'
              << "  错误说明: " << cudaGetErrorString(result) << '\n'
              << "  代码位置: " << file << ':' << line << std::endl;

    // CUDA 错误后程序状态可能已不再可信，因此立即以失败状态退出。
    std::exit(EXIT_FAILURE);
}

}  // namespace cuda_foundations

// 宏保留调用点的源文件和行号，同时保证传入的 CUDA 表达式只求值一次。
#define CUDA_CHECK(expression) \
    ::cuda_foundations::check_cuda((expression), #expression, __FILE__, __LINE__)
