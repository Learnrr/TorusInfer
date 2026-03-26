#include "layer/ResidualAdd.h"
#include "kernel/residual_add_kernel.h"

void ResidualAdd::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    const size_t num_elements = input.numel();
    (void)context;
    launch_residual_add_kernel(
        output.data,
        input.data,
        output.data,
        num_elements,
        input.dtype
    );
}

void ResidualAdd::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    prefill_forward(input, output, context);
}