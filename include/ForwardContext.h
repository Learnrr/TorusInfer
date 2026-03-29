#pragma once

#include "Batch.h"
#include "Workspace.h"
#include "llm_engine_config.h"
struct ForwardContext {
    size_t layer_id;
    Batch* batch;
    Workspace* workspace;
    LLMEngineConfig* config;
};