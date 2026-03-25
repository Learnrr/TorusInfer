#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include "embedding_kernel.h"

template <typename T>
__global__ void embedding_kernel(
    const size_t* input, 
    const T* embedding_table,
    T* output,
    size_t batch_seq_len,
    size_t hidden_size
){
    int token = blockIdx.x;
    int h = threadIdx.x;
    if(token < batch_seq_len && h < hidden_size) {
        size_t token_id = input[token];
        T val = embedding_table[token_id * hidden_size + h];
        output[token * hidden_size + h] = val;
    }

}

void launch_embedding_kernel(
    const size_t* input, 
    const void* embedding_table,
    void* output,
    size_t batch_seq_len,
    size_t hidden_size,
    DataType dtype
) {
    if (dtype == DataType::FLOAT32) {
        embedding_kernel<float><<<batch_seq_len, hidden_size>>>(
            input,
            static_cast<const float*>(embedding_table),
            static_cast<float*>(output),
            batch_seq_len,
            hidden_size
        );
    } else {
        embedding_kernel<__half><<<batch_seq_len, hidden_size>>>(
            input,
            static_cast<const __half*>(embedding_table),
            static_cast<__half*>(output),
            batch_seq_len,
            hidden_size
        );
    }
}