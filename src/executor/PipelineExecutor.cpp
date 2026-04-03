#include "executor/PiplineExecutor.h"

void PiplineExecutor::run_prefill(Batch& batch) {
    run_prefill(batch, nullptr);
}

void PiplineExecutor::run_decode(Batch& batch) {
    run_decode(batch, nullptr);
}


void PiplineExecutor::run_prefill(Batch& batch, void* external_hidden_in) {
    model->stage_prefill_forward(
        batch,
        *workspace,
        stage_start_layer,
        stage_end_layer,
        external_hidden_in,
        seq_pool
    );
}

void PiplineExecutor::run_decode(Batch& batch, void* external_hidden_in) {
    model->stage_decode_forward(
        batch,
        *workspace,
        stage_start_layer,
        stage_end_layer,
        external_hidden_in,
        seq_pool
    );
}