#include "residual_add_kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__device__ inline float to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ inline float to_float<__half>(__half v) {
    return __half2float(v);
}

template <typename T>
__device__ inline T from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ inline __half from_float<__half>(float v) {
    return __float2half(v);
}

template <typename T>
__global__ void residual_add_kernel(
    const T* residual,
    const T* input,
    T* output,
    size_t num_elements
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const float sum = to_float<T>(residual[idx]) + to_float<T>(input[idx]);
    output[idx] = from_float<T>(sum);
}

void launch_residual_add_kernel(
    const void* residual,
    const void* input,
    void* output,
    size_t num_elements,
    DataType dtype
) {
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((num_elements + threads_per_block - 1) / threads_per_block);
    if (dtype == DataType::FLOAT32) {
        residual_add_kernel<float><<<blocks, threads_per_block>>>(
            static_cast<const float*>(residual),
            static_cast<const float*>(input),
            static_cast<float*>(output),
            num_elements
        );
    } else {
        residual_add_kernel<__half><<<blocks, threads_per_block>>>(
            static_cast<const __half*>(residual),
            static_cast<const __half*>(input),
            static_cast<__half*>(output),
            num_elements
        );
    }
}
