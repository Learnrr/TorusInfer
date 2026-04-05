#pragma once

#include "executor/Executor.h"
#include "model/IModel.h"
#include "SequencePool.h"
#include "Workspace.h"
#include "Batch.h"
#include "model/ModelForwardContext.h"

class SingleCardExecutor : public Executor {
public:
    SingleCardExecutor(
        IModel* model, 
        Workspace* workspace, 
        SequencePool* seq_pool = nullptr
    )
        : model(model), 
        workspace(workspace), 
        seq_pool(seq_pool) {}

    ErrorCode run_prefill(Batch& batch, ModelForwardContext& context) override;
    ErrorCode run_decode(Batch& batch, ModelForwardContext& context) override;

    bool poll_completion(CompletionRecord& out_record) override;

    void run_release_events(Batch& batch) override;
    void run_stop() override;
    void run_free(Batch& batch) override;    

private:
    IModel* model;
    Workspace* workspace;
    SequencePool* seq_pool;
};