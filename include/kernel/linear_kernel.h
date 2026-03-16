#pragma once

#include <cstddef>

void launch_mlp_linear_kernel(
    const float* input,
    const float* weight,
    float* output,
    size_t batch_seq_len,
    size_t in_features,
    size_t out_features
);
