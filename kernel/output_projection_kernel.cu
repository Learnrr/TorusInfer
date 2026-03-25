
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include "output_projection_kernel.h"

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
 __global__ void output_projection_kernel(
    const T* input,
    const T* weight,
    T* output,
    size_t batch_seq_len,
    size_t num_heads,
    size_t head_dim
) {
    int token = blockIdx.x;
    int head = blockIdx.y;
    int dim = threadIdx.x;
    if(token < batch_seq_len && head < num_heads && dim < head_dim){
        int in_features = num_heads * head_dim;
        int out_features = in_features;
        int out_col = head * head_dim + dim;
        float sum = 0.0f;
        for (int in_d = 0; in_d < in_features; ++in_d) {
            float in_val = to_float<T>(input[token * in_features + in_d]);
            float w_val = to_float<T>(weight[in_d * out_features + out_col]);
            sum += in_val * w_val;
        }
        output[token * in_features + out_col] = from_float<T>(sum);
    }
}

void launch_output_projection_kernel(
    const void* input,
    const void* weight,
    void* output,
    size_t batch_seq_len,
    size_t num_heads,
    size_t head_dim,
    DataType dtype
) {
    dim3 grid(batch_seq_len, num_heads);
    dim3 block(head_dim);
    if (dtype == DataType::FLOAT32) {
        output_projection_kernel<float><<<grid, block>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(weight),
            static_cast<float*>(output),
            batch_seq_len,
            num_heads,
            head_dim
        );
    } else {
        output_projection_kernel<__half><<<grid, block>>>(
            static_cast<const __half*>(input),
            static_cast<const __half*>(weight),
            static_cast<__half*>(output),
            batch_seq_len,
            num_heads,
            head_dim
        );
    }
}