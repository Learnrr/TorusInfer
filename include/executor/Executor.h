#pragma once
#include "Batch.h"
#include "model/ModelForwardContext.h"
#include "error.h"
#include "channel/ChannelMessage.h"

enum class CompletionStatus {
    DONE = 0,
    INVALID = 1,
    RELEASE_EVENTS_FAILED = 2,
    UNKNOWN = 3,
};

struct CompletionRecord {
    size_t batch_id = 0;
    ForwardOp op_type = ForwardOp::UNKNOWN;
    CompletionStatus status = CompletionStatus::UNKNOWN;
    std::vector<size_t> sampled_token_ids;
    std::vector<size_t> sequence_ids;
};

class Executor {
    public:
    virtual ~Executor() = default;
    virtual ErrorCode run_prefill(Batch& batch, ModelForwardContext& context) = 0;
    virtual ErrorCode run_decode(Batch& batch, ModelForwardContext& context) = 0;

    virtual bool poll_completion(CompletionRecord& out_record) = 0;

    virtual void run_release_events(Batch& batch) = 0;
    virtual void run_stop() = 0;
    virtual void run_free(Batch& batch) = 0;
};