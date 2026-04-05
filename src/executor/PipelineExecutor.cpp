#include "executor/PiplineExecutor.h"
#include "model/ModelForwardContext.h"

ErrorCode PiplineExecutor::run_prefill(Batch& batch, ModelForwardContext& context) {
    model->stage_prefill_forward(batch, context);
    return ErrorCode::SUCCESS;
}

ErrorCode PiplineExecutor::run_decode(Batch& batch, ModelForwardContext& context) {
    model->stage_decode_forward(batch, context);
    return ErrorCode::SUCCESS;
}

bool PiplineExecutor::poll_completion(CompletionRecord& out_record) {
    (void)out_record;
    return false;
}

void PiplineExecutor::run_release_events(Batch& batch) {
    (void)batch;
    // pipeline executor does not have events to release, just return
}

void PiplineExecutor::run_stop() {
    // pipeline executor does not have a receive thread to stop, just return
}

void PiplineExecutor::run_free(Batch& batch) {
    (void)batch;
    // pipeline executor does not have extra cache to free, just return
}