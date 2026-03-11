#pragma once
#include "CacheBlock.h"
#include "define.h"
#include <cuda_runtime.h>
#include "Sequence.h"

class KVCacheManager {
    private:
        void init();
        void* key_cache;
        void* value_cache;

        vector<CacheBlock> free_blocks;
        vector<CacheBlock> used_blocks;
    public:
        KVCacheManager(){
            init();
        }
        
        CacheBlock* get_cache_block(size_t block_id);


        CacheBlock* allocate_cache_block();

        void free_cache_block(size_t block_id);

        ~KVCacheManager() {
            cudaFree(key_cache);
            cudaFree(value_cache);
        }
    private:


        size_t num_blocks;
        size_t block_size;
};