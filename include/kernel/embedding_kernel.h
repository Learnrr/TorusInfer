#pragma once

#include <cstddef>
#include "define.h"

void launch_embedding_kernel(
    const size_t* input,
    const void* embedding_table,
    void* output,
    size_t batch_seq_len,
    size_t hidden_size,
    DataType dtype
);
