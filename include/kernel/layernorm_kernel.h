#pragma once

#include <cstddef>
#include "define.h"

void launch_layernorm_kernel(
    const void* input,
    const void* gamma,
    void* output,
    size_t num_tokens,
    size_t hidden_size,
    float eps,
    DataType dtype
);
