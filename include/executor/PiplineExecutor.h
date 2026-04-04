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

    void run_prefill(Batch& batch, ModelForwardContext& context) override;
    void run_decode(Batch& batch, ModelForwardContext& context) override;

private:
    IModel* model;
    Workspace* workspace;
    size_t stage_start_layer;
    size_t stage_end_layer;
    SequencePool* seq_pool;
};