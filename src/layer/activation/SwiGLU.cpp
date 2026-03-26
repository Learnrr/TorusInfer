#include "layer/activation/SwiGLU.h"
#include "kernel/swiglu_kernel.h"

void SwiGLU::forward(
    const Tensor& gate,
    const Tensor& up,
    Tensor& output,
    ForwardContext& context
) {
    if (gate.data == nullptr 
        || up.data == nullptr 
        || output.data == nullptr 
        || context.batch == nullptr) {
        return;
    }

    const size_t num_tokens = context.batch->num_tokens;

    launch_swiglu_kernel_from_gate_up(
        gate.data,
        up.data,
        output.data,
        num_tokens,
        hidden_size,
        gate.dtype
    );
}