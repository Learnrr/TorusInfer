
#include "cuda_runtime.h"

 __global__ void output_projection_kernel(
    const float* input,
    const float* weight,
    float* output,
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
            float in_val = input[token * in_features + in_d];
            float w_val = weight[in_d * out_features + out_col];
            sum += in_val * w_val;
        }
        output[token * in_features + out_col] = sum;
    }
}