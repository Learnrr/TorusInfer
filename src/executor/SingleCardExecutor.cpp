#include "executor/SingleCardExecutor.h"
#include "utils/logger.h"

ErrorCode SingleCardExecutor::run_prefill(Batch& batch, ModelForwardContext& context) {
    if (context.workspace == nullptr) {
        return ErrorCode::INVALID_INPUT;
    }
    model->prefill_forward(batch, context);
    return ErrorCode::SUCCESS;
}

ErrorCode SingleCardExecutor::run_decode(Batch& batch, ModelForwardContext& context) {
    if (context.workspace == nullptr) {
        return ErrorCode::INVALID_INPUT;
    }
    model->decode_forward(batch, context);
    return ErrorCode::SUCCESS;
}

ErrorCode SingleCardExecutor::run_free(Batch& batch) {
    if (seq_pool == nullptr || cache_manager == nullptr) {
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    for (size_t seq_id : batch.sequence_ids) {
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }

        for (const auto& block : seq->blocks) {
            if (block) {
                cache_manager->free_cache_block(block->block_id);
            }
        }
        seq->blocks.clear();
        seq_pool->erase(seq_id);
    }

    return ErrorCode::SUCCESS;
}

ErrorCode SingleCardExecutor::run_release_events(Batch& batch) {
    if (retained_outgoing_events == nullptr) {
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    auto it = retained_outgoing_events->find(batch.batch_id);
    if (it == retained_outgoing_events->end()) {
        return ErrorCode::SUCCESS;
    }

    cudaEvent_t event_to_release = it->second;
    retained_outgoing_events->erase(it);
    cudaError_t destroy_err = cudaEventDestroy(event_to_release);
    if (destroy_err != cudaSuccess) {
        LOG_ERROR("SingleCardExecutor failed to destroy retained outgoing event: " + std::string(cudaGetErrorString(destroy_err)));
        return ErrorCode::CUDA_FAILURE;
    }

    return ErrorCode::SUCCESS;
}

ErrorCode SingleCardExecutor::run_stop() {
    // single card executor does not have a receive thread to stop, just return
    return ErrorCode::SUCCESS;
}