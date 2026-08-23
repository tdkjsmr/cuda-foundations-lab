#pragma once

#include "cuda_check.cuh"

#include <cuda_runtime.h>

namespace cuda_foundations {

// CudaEventTimer 用一对 CUDA Event 测量同一 Stream 上提交的 GPU 工作。
class CudaEventTimer {
public:
    // 构造时创建 Event，避免把 Event 创建成本混入正式计时区间。
    CudaEventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    // Event 是独占资源，禁止复制可避免同一个句柄被销毁两次。
    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    // 析构函数释放两个 Event，并继续执行统一 CUDA 返回值检查。
    ~CudaEventTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }

    // 在指定 Stream 当前队尾记录计时起点。
    void start(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    // 在同一 Stream 记录终点并仅等待 stop Event，返回区间的毫秒数。
    float stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

}  // namespace cuda_foundations
