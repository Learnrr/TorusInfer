#pragma once

#include <cstddef>
#include "define.h"

void launch_residual_add_kernel(
    const void* residual,
    const void* input,
    void* output,
    size_t num_elements,
    DataType dtype
);
