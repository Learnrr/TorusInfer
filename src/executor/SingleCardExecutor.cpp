#include "executor/SingleCardExecutor.h"

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

void SingleCardExecutor::run_free(Batch& batch) {
    (void)batch;
    // single card executor does not have extra cache to free, just return
}

void SingleCardExecutor::run_release_events(Batch& batch) {
    (void)batch;
    // single card executor does not have events to release, just return
}

void SingleCardExecutor::run_stop() {
    // single card executor does not have a receive thread to stop, just return
}

bool SingleCardExecutor::poll_completion(CompletionRecord& out_record) {
    (void)out_record;
    return false;
}