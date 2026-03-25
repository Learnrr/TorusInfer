#pragma once

#include <cstddef>
#include "define.h"

void launch_mlp_linear_kernel(
    const void* input,
    const void* weight,
    void* output,
    size_t batch_seq_len,
    size_t in_features,
    size_t out_features,
    DataType dtype
);
