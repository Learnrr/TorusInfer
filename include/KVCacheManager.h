#pragma once
#include "CacheBlock.h"
#include "define.h"
#include <cuda_runtime.h>
#include "Sequence.h"
#include "utils/include/error.h"
#include <variant>

class KVCacheManager {
    private:
        ErrorCode init();
        void* key_cache;
        void* value_cache;

        vector<shared_ptr<CacheBlock>> free_blocks;
        vector<shared_ptr<CacheBlock>> used_blocks;
    public:
        KVCacheManager(){};
        
        variant<shared_ptr<CacheBlock>, ErrorCode> get_cache_block(size_t block_id);

        variant<shared_ptr<CacheBlock>, ErrorCode> allocate_cache_block();

        ErrorCode free_cache_block(size_t block_id);

        ~KVCacheManager() {
            cudaFree(key_cache);
            cudaFree(value_cache);
        }
    private:


        size_t num_blocks;
        size_t block_size;
};