#pragma once

#include <cstddef>
#include "define.h"

void launch_write_kvcache_kernel(
    void** kcache_block_ptrs,
    void** vcache_block_ptrs,
    const size_t* block_ids,
    const size_t* block_offsets,
    const void* key_data,
    const void* value_data,
    int num_tokens,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int layer_id,
    DataType dtype
);
