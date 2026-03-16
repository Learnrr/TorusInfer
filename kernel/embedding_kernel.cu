#include "cuda_runtime.h"
#include "embedding_kernel.h"

__global__ void embedding_kernel(
    const size_t* input, 
    float* embedding_table,
    float* output,
    size_t batch_seq_len,
    size_t hidden_size
){
    // Implement the embedding kernel logic here
    int token = blockIdx.x;
    int h = threadIdx.x;
    if(token < batch_seq_len && h < hidden_size) {
        size_t token_id = input[token];
        float val = embedding_table[token_id * hidden_size + h];
        output[token * hidden_size + h] = val;
    }

}

void launch_embedding_kernel(
    const size_t* input, 
    float* embedding_table,
    float* output,
    size_t batch_seq_len,
    size_t hidden_size
) {
    // token maps to blockIdx.x, hidden dim maps to threadIdx.x.
    embedding_kernel<<<batch_seq_len, hidden_size>>>(
        input,
        embedding_table,
        output,
        batch_seq_len,
        hidden_size
    );
}