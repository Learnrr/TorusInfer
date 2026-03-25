#pragma once

#include <cstddef>
#include "define.h"

void launch_output_projection_kernel(
    const void* input,
    const void* weight,
    void* output,
    size_t batch_seq_len,
    size_t num_heads,
    size_t head_dim,
    DataType dtype
);
