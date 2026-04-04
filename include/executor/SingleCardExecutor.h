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

    void run_prefill(Batch& batch, ModelForwardContext& context) override;
    void run_decode(Batch& batch, ModelForwardContext& context) override;

private:
    IModel* model;
    Workspace* workspace;
    SequencePool* seq_pool;
};