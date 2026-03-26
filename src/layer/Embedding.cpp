#include "layer/Embedding.h"
#include "kernel/embedding_kernel.h"
#include "cuda_runtime.h"
#include "utils/cuda_deleter.h"
#include "utils/logger.h"

void Embedding::forward(const std::vector<size_t>& token_ids, Tensor& output, size_t num_tokens) {
    if (num_tokens == 0) {
        LOG_ERROR("Embedding::forward got num_tokens=0");
        return;
    }
    if (token_ids.size() < num_tokens) {
        LOG_ERROR("Embedding::forward token_ids smaller than num_tokens");
        return;
    }
    if (embedding_weight_gpu == nullptr || embedding_weight_gpu->data == nullptr) {
        LOG_ERROR("Embedding::forward embedding weight is null");
        return;
    }
    if (output.data == nullptr) {
        LOG_ERROR("Embedding::forward output tensor data is null");
        return;
    }

    //put token_ids to gpu
    size_t* d_token_ids_raw = nullptr;
    cudaError_t cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_token_ids_raw), num_tokens * sizeof(size_t));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Embedding::forward cudaMalloc failed for token ids");
        return;
    }
    CudaUniquePtr<size_t> d_token_ids(d_token_ids_raw);

    cuda_err = cudaMemcpy(d_token_ids.get(), token_ids.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Embedding::forward cudaMemcpy failed for token ids");
        return;
    }

    launch_embedding_kernel(
        d_token_ids.get(), 
        embedding_weight_gpu->data,
        output.data,
        num_tokens, 
        embedding_dim,
        output.dtype
    );

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Embedding::forward kernel launch failed");
        return;
    }
}
