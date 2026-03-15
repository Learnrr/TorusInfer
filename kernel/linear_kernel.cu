
__global__ void mlp_linear_kernel(
    const float* input,      // [batch_seq_len, in_features]
    const float* weight,     // [in_features, out_features]
    float* output,           // [batch_seq_len, out_features]
    size_t batch_seq_len,
    size_t in_features,
    size_t out_features
){
    int token = blockIdx.x;
    int out_col = blockIdx.y * blockDim.x + threadIdx.x;
    if(token < batch_seq_len && out_col < out_features){
        float sum = 0.0f;
        for(int in_d = 0; in_d < in_features; ++in_d){
            float in_val = input[token * in_features + in_d];
            float w_val = weight[in_d * out_features + out_col];
            sum += in_val * w_val;
        }
        output[token * out_features + out_col] = sum;
    }
}