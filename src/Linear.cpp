#include "Linear.h"
#include "cuda_runtime.h"

extern __global__ void mlp_linear_kernel(
    const float* input,
    const float* weight,
    float* output,
    size_t batch_seq_len,
    size_t in_features,
    size_t out_features
);

void Linear::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    run_linear(input, output, context);
}

void Linear::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    run_linear(input, output, context);
}

void Linear::run_linear(const Tensor& input, Tensor& output, ForwardContext& context) {

    size_t batch_seq_len = context.batch->num_tokens;
    dim3 block(256);
    dim3 grid(batch_seq_len, (config.out_features + block.x - 1) / block.x);
    mlp_linear_kernel<<<grid, block>>>(
        static_cast<const float*>(input.data),
        static_cast<const float*>(linear_weight.data),
        static_cast<float*>(output.data),
        batch_seq_len,
        config.in_features,
        config.out_features
    );

}