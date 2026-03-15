#include "cuda_runtime.h"
__global__ void write_kvcache_kernel(
    float** kcache_block_ptrs,
    float** vcache_block_ptrs,
    const size_t* block_ids,
    const size_t* block_offsets,
    const float* key_data,
    const float* value_data,
    int num_tokens,
    int num_layers,
    int num_heads,
    int head_dim,
    int block_size,
    int layer_id
){
    int token = blockIdx.x;
    int head = blockIdx.y;
    int dim = threadIdx.x;
    if (token >= num_tokens || head >= num_heads || dim >= head_dim) return;

    int kv_offset = token * num_heads * head_dim + head * head_dim + dim;
    float k_val = key_data[kv_offset];
    float v_val = value_data[kv_offset];

    size_t blk = block_ids[token];
    size_t off = block_offsets[token];

    float* k_block = kcache_block_ptrs[blk];
    float* v_block = vcache_block_ptrs[blk];

    int cache_offset = off * num_layers * num_heads * head_dim + layer_id * num_heads * head_dim + head * head_dim + dim;
    k_block[cache_offset] = k_val;
    v_block[cache_offset] = v_val;
}