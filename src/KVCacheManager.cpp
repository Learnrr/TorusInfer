#include "KVCacheManager.h"

void KVCacheManager::init(){
    
    size_t total_cache_size = 
        BLOCK_SIZE * NUM_LAYERS * HEAD_DIM * NUM_HEADS * DTYPE;

    num_blocks = total_cache_size / BLOCK_SIZE;

    cudaError_t error = cudaMalloc(&key_cache, total_cache_size);
    if (error != cudaSuccess) {
        // Handle error
    }
    error = cudaMalloc(&value_cache, total_cache_size);
    if (error != cudaSuccess) {
        // Handle error
    }
    for (size_t i = 0; i < num_blocks; ++i) {
        CacheBlock block(
            i, 
            key_cache + i * (BLOCK_SIZE*NUM_LAYERS*HEAD_DIM*NUM_HEADS*DTYPE),
            value_cache + i * (BLOCK_SIZE*NUM_LAYERS*HEAD_DIM*NUM_HEADS*DTYPE)
        );
        free_blocks.push_back(block);
    }
}

CacheBlock* KVCacheManager::get_cache_block(size_t block_id) {
    if (block_id >= num_blocks) return nullptr;
    return &used_blocks[block_id];
}


CacheBlock* KVCacheManager::allocate_cache_block() {
    if (free_blocks.empty()) {
        return nullptr;
    }
    CacheBlock block = free_blocks.back();
    free_blocks.pop_back();
    used_blocks.push_back(block);
    return &used_blocks.back();
}

void KVCacheManager::free_cache_block(size_t block_id) {
    if (block_id >= num_blocks) return;
    auto it = std::find_if(used_blocks.begin(), used_blocks.end(), 
        [block_id](const CacheBlock& block) { return block.block_id == block_id; });
    if (it != used_blocks.end()) {
        free_blocks.push_back(*it);
        used_blocks.erase(it);
    }
}