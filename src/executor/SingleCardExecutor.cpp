#include "executor/SingleCardExecutor.h"

void SingleCardExecutor::run_prefill(Batch& batch, ModelForwardContext& context) {
    if (context.workspace == nullptr) {
        return;
    }
    model->prefill_forward(batch, context);
}

void SingleCardExecutor::run_decode(Batch& batch, ModelForwardContext& context) {
    if (context.workspace == nullptr) {
        return;
    }
    model->decode_forward(batch, context);
}