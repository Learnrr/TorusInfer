#include "layer/Linear.h"
#include "cuda_runtime.h"
#include "kernel/linear_kernel.h"

void Linear::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    size_t batch_seq_len = context.batch->num_tokens;
    launch_mlp_linear_kernel(
        input.data,
        linear_weight.data,
        output.data,
        batch_seq_len,
        config.in_features,
        config.out_features,
        input.dtype
    );
}

void Linear::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    size_t batch_seq_len = context.batch->num_tokens;
    launch_mlp_linear_kernel(
        input.data,
        linear_weight.data,
        output.data,
        batch_seq_len,
        config.in_features,
        config.out_features,
        input.dtype
    );
}
