#pragma once
#include "Batch.h"

class Executor {
    public:
    virtual ~Executor() = default;
    virtual void run_prefill(Batch& batch ) = 0;
    virtual void run_decode(Batch& batch) = 0;
};

