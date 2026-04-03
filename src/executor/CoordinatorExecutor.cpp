#include "executor/CoordinatorExecutor.h"

#include "utils/logger.h"

namespace {
void dispatch_to_worker(
    Channel* output,
    ForwardOp op,
    Batch& batch
) {
    if (output == nullptr) {
        LOG_ERROR("CoordinatorExecutor output channel is null");
        return;
    }

    ForwardMessage message;
    message.op_type = op;
    message.batch = batch;
    output->send(message);
}

void receive_from_worker(Channel* input, Batch& batch) {
    if (input == nullptr) {
        LOG_ERROR("CoordinatorExecutor input channel is null.");
        return;
    }

    ForwardMessage response;
    input->receive(response);
    // Worker returns sampled tokens in batch.token_ids for done:decode/done:prefill.
    if (!response.batch.token_ids.empty()) {
        batch.sampled_token_ids = response.batch.token_ids;
    }
}




void CoordinatorExecutor::run_prefill(Batch& batch) {
    dispatch_to_worker(to_worker0, ForwardOp::PREFILL, batch);
    receive_from_worker(from_worker_last, batch);
}

void CoordinatorExecutor::run_decode(Batch& batch) {
    dispatch_to_worker(to_worker0, ForwardOp::DECODE, batch);
    receive_from_worker(from_worker_last, batch);
}

void CoordinatorExecutor::run_free(Batch& batch) {
    dispatch_to_worker(to_worker0, ForwardOp::FREE_SEQ, batch);

    ForwardMessage response;
    from_worker_last->receive(response);
    if (response.op_type != ForwardOp::DONE) {
        LOG_ERROR("CoordinatorExecutor expected DONE response for FREE_SEQ");
        return;
    }
}