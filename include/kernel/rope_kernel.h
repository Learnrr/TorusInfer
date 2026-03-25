#pragma once

#include <cstddef>
#include "define.h"

void launch_apply_rope_inplace(
    void* tensor,
    const size_t* positions,
    size_t num_tokens,
    size_t num_heads,
    size_t head_dim,
    float rope_theta,
    DataType dtype
);
