#pragma once

#include "executor/Executor.h"
#include "model/IModel.h"
#include "model/ModelForwardContext.h"
#include "SequencePool.h"
#include "Workspace.h"

class PiplineExecutor : public Executor {
public:
    PiplineExecutor(
        IModel* model, 
        Workspace* workspace, 
        size_t stage_start_layer, 
        size_t stage_end_layer, 
        SequencePool* seq_pool = nullptr
    )
        : model(model), 
        workspace(workspace), 
        stage_start_layer(stage_start_layer), 
        stage_end_layer(stage_end_layer), 
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
    size_t stage_start_layer;
    size_t stage_end_layer;
    SequencePool* seq_pool;
};