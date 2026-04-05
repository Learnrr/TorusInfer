#include "executor/PiplineExecutor.h"
#include "model/ModelForwardContext.h"

void PiplineExecutor::run_prefill(Batch& batch, ModelForwardContext& context) {
    model->stage_prefill_forward(batch, context);
}

void PiplineExecutor::run_decode(Batch& batch, ModelForwardContext& context) {
    model->stage_decode_forward(batch, context);
}