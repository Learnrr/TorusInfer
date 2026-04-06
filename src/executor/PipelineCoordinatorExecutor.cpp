#include "executor/PipelineCoordinatorExecutor.h"
#include "utils/logger.h"

namespace {
void dispatch_to_worker(
    Channel* output,
    ForwardOp op,
    const Batch& batch
) {
    if (output == nullptr) {
        LOG_ERROR("PipelineCoordinatorExecutor output channel is null");
        return;
    }

    ForwardMessage message;
    message.op_type = op;
    message.batch = batch;
    output->send(message);
}
} // namespace

PipelineCoordinatorExecutor::~PipelineCoordinatorExecutor() {
    stop_receive_thread.store(true);
    if (receive_thread.joinable()) {
        // Try to nudge the pipeline to unblock receiver if it is waiting.
        if (to_worker0 != nullptr) {
            Batch control_batch;
            dispatch_to_worker(to_worker0, ForwardOp::STOP, control_batch);
        }
        receive_thread.join();
    }
}

void PipelineCoordinatorExecutor::start_receive_thread() {
    if (receiver_started.load()) {
        return;
    }
    if (from_worker_last == nullptr) {
        return;
    }

    stop_receive_thread.store(false);
    receiver_started.store(true);
    receive_thread = std::thread([this]() {
        while (!stop_receive_thread.load()) {
            if (!receive_and_track()) {
                if (!stop_receive_thread.load()) {
                    LOG_ERROR("PipelineCoordinatorExecutor receive loop failed; stopping receiver thread.");
                }
                break;
            }
        }
    });
}

bool PipelineCoordinatorExecutor::receive_and_track() {
    if (from_worker_last == nullptr) {
        LOG_ERROR("PipelineCoordinatorExecutor input channel is null.");
        return false;
    }

    ForwardMessage response;
    from_worker_last->receive(response);

    // FREE_SEQ control acknowledgements do not correspond to inflight batches.
    if (response.op_type == ForwardOp::DONE && response.batch.batch_id == 0 && !response.batch.sequence_ids.empty()) {
        return true;
    }

    CompletionRecord record;
    record.batch_id = response.batch.batch_id;
    record.op_type = response.op_type;
    record.sequence_ids = response.batch.sequence_ids;
    record.sampled_token_ids = response.batch.sampled_token_ids;

    bool ok = true;
    if (response.op_type == ForwardOp::DONE) {
        record.status = CompletionStatus::DONE;
    } else if (response.op_type == ForwardOp::INVALID) {
        record.status = CompletionStatus::INVALID;
        ok = false;
    } else if (response.op_type == ForwardOp::RELEASE_EVENTS_FAILED) {
        record.status = CompletionStatus::RELEASE_EVENTS_FAILED;
        ok = false;
    } else {
        record.status = CompletionStatus::UNKNOWN;
        ok = false;
    }

    {
        std::lock_guard<std::mutex> lock(completion_mutex);
        completed_records.push_back(std::move(record));
    }

    return ok;
}

ErrorCode PipelineCoordinatorExecutor::run_prefill(Batch& batch, ModelForwardContext& context) {
    (void)context;
    if (to_worker0 == nullptr) {
        LOG_ERROR("PipelineCoordinatorExecutor cannot submit PREFILL: output channel is null");
        return ErrorCode::INITIANLIZATION_ERROR;
    }
    submit_prefill_batch(batch);
    return ErrorCode::SUCCESS;
}

ErrorCode PipelineCoordinatorExecutor::run_decode(Batch& batch, ModelForwardContext& context) {
    (void)context;
    if (to_worker0 == nullptr) {
        LOG_ERROR("PipelineCoordinatorExecutor cannot submit DECODE: output channel is null");
        return ErrorCode::INITIANLIZATION_ERROR;
    }
    submit_decode_batch(batch);
    return ErrorCode::SUCCESS;
}

ErrorCode PipelineCoordinatorExecutor::run_free(Batch& batch) {
    if (to_worker0 == nullptr) {
        LOG_ERROR("PipelineCoordinatorExecutor cannot submit FREE_SEQ: output channel is null");
        last_forward_ok = false;
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    dispatch_to_worker(to_worker0, ForwardOp::FREE_SEQ, batch);
    last_forward_ok = true;
    return ErrorCode::SUCCESS;
}

ErrorCode PipelineCoordinatorExecutor::run_release_events(Batch& batch) {
    if (to_worker0 == nullptr) {
        LOG_ERROR("PipelineCoordinatorExecutor cannot submit RELEASE_EVENTS: output channel is null");
        last_forward_ok = false;
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    dispatch_to_worker(to_worker0, ForwardOp::RELEASE_EVENTS, batch);
    // The dedicated receiver thread is the only consumer of from_worker_last.
    last_forward_ok = true;
    return ErrorCode::SUCCESS;
}

ErrorCode PipelineCoordinatorExecutor::run_stop() {
    if (to_worker0 == nullptr) {
        LOG_ERROR("PipelineCoordinatorExecutor cannot submit STOP: output channel is null");
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    Batch control_batch;
    dispatch_to_worker(to_worker0, ForwardOp::STOP, control_batch);
    return ErrorCode::SUCCESS;
}

bool PipelineCoordinatorExecutor::consume_last_forward_ok() {
    const bool ok = last_forward_ok;
    last_forward_ok = true;
    return ok;
}

void PipelineCoordinatorExecutor::submit_decode_batch(const Batch& batch) {
    dispatch_to_worker(to_worker0, ForwardOp::DECODE, batch);
}

void PipelineCoordinatorExecutor::submit_prefill_batch(const Batch& batch) {
    dispatch_to_worker(to_worker0, ForwardOp::PREFILL, batch);
}

bool PipelineCoordinatorExecutor::poll_completion(CompletionRecord& out_record) {
    std::lock_guard<std::mutex> lock(completion_mutex);
    if (completed_records.empty()) {
        return false;
    }
    out_record = std::move(completed_records.front());
    completed_records.pop_front();
    return true;
}
