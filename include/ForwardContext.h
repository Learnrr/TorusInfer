#pragma once

#include "Batch.h"
#include "SequencePool.h"
#include "Workspace.h"
#include "llm_engine_config.h"
struct ForwardContext {
    size_t layer_id;
    Batch* batch;
    SequencePool* seq_pool = nullptr;
    Workspace* workspace;
    LLMEngineConfig* config;
};