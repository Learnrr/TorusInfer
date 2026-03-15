#include "Embedding.h"

void Embedding::forward(size_t* token_ids, Tensor& output, size_t num_tokens, size_t embedding_dim) {
    size_t* d_token_ids;
    cudaMalloc(&d_token_ids, num_tokens * sizeof(size_t));
    cudaMemcpy(d_token_ids, token_ids, num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    embedding_kernel<<<num_tokens, embedding_dim>>>(
        d_token_ids, 
        layer_layout->embedding_weights.data, 
        output.data, 
        num_tokens, 
        embedding_dim
    );
    
}
