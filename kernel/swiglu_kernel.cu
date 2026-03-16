#include "cuda_runtime.h"
#include "swiglu_kernel.h"

__global__ void swiglu_kernel(
    const float* input,
    float* output,
    size_t hidden_size,
    size_t total_elements
){
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    size_t token = idx / hidden_size;
    size_t dim = idx % hidden_size;

    const size_t token_input_base = token * (hidden_size * 2);
    const float gate = input[token_input_base + dim];
    const float up = input[token_input_base + hidden_size + dim];

    // SwiGLU: SiLU(gate) * up
    const float sigmoid_gate = 1.0f / (1.0f + __expf(-gate));
    output[idx] = (gate * sigmoid_gate) * up;
}

void launch_swiglu_kernel(
    const float* input,
    float* output,
    size_t num_tokens,
    size_t hidden_size
) {
    const size_t total_elements = num_tokens * hidden_size;
    if (total_elements == 0) {
        return;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    swiglu_kernel<<<blocks, threads>>>(
        input,
        output,
        hidden_size,
        total_elements
    );
}