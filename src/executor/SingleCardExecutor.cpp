#include "executor/SingleCardExecutor.h"

void SingleCardExecutor::run_prefill(Batch& batch, Workspace* workspace) {
    model->prefill_forward(batch, *workspace, seq_pool);
}

void SingleCardExecutor::run_decode(Batch& batch, Workspace* workspace) {
    model->decode_forward(batch, *workspace, seq_pool);
}