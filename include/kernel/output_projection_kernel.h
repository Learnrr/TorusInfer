#pragma once

#include <cstddef>

void launch_output_projection_kernel(
    const float* input,
    const float* weight,
    float* output,
    size_t batch_seq_len,
    size_t num_heads,
    size_t head_dim
);
