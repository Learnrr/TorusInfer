#pragma once
#include "Batch.h"
#include "model/ModelForwardContext.h"

class Executor {
    public:
    virtual ~Executor() = default;
    virtual void run_prefill(Batch& batch, ModelForwardContext& context) = 0;
    virtual void run_decode(Batch& batch, ModelForwardContext& context) = 0;
};

