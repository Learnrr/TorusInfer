#include "cuda_runtime.h"

__global__ void projection_kernel(
    const void* input, 
    const void* weight, 
    const void* bias, 
    void* output,
    size_t batch_seq_len,
    size_t num_attention_heads,
    size_t head_dim
){
    // Implement the projection kernel logic here
    int token = blockId.x;
    int head = blockId.y;
    int dim = threadIdx.x;
    int qkv = blockId.z; // 0 for q, 1 for k, 2 for v

    if(token < batch_seq_len && head < num_attention_heads && dim < head_dim){
        int in_offset = token * num_attention_heads * head_dim + head * head_dim + dim;
        int w_offset = (head * head_dim + dim) * num_attention_heads * head_dim * 3 + qkv * num_attention_heads * head_dim + head * head_dim + dim;
        int out_offset = token * num_attention_heads * head_dim * 3 + qkv * num_attention_heads * head_dim + head * head_dim + dim;

        float sum = 0.0f;
        int in_features = num_attention_heads * head_dim;
        int out_features = 3 * in_features;
        int out_col = qkv * in_features + head * head_dim + dim;
        for (int in_d = 0; in_d < in_features; ++in_d) {
            float in_val = ((float*)input)[token * in_features + in_d];
            float w_val = ((float*)weight)[in_d * out_features + out_col];
            sum += in_val * w_val;
        }
        output[out_offset] = sum;
    }

}