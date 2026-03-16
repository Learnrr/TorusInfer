#pragma once

#include <cstddef>

void launch_embedding_kernel(
    const size_t* input,
    float* embedding_table,
    float* output,
    size_t batch_seq_len,
    size_t hidden_size
);
