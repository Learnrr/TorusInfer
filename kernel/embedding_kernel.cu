#include "cuda_runtime.h"
__global__ void embedding_kernel(
    const size_t* input, 
    void* embedding_table,
    void* output,
    size_t batch_seq_len,
    size_t hidden_size
){
    // Implement the embedding kernel logic here
    int token = blockIdx.x;
    int h = threadIdx.x;
    if(token < batch_seq_len && h < hidden_size) {
        size_t token_id = input[token];
        float val = ((float*)embedding_table)[token_id * hidden_size + h];
        ((float*)output)[token * hidden_size + h] = val;
    }

}