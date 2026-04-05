#pragma once

#include "model/ModelForwardContext.h"
#include "llm_engine_config.h"
#include "Batch.h"
#include <memory>
#include <string>

class IModel{
    public:
    virtual void init(LLMEngineConfig& config) = 0;
        virtual void prefill_forward(Batch& batch, ModelForwardContext& context) = 0;
        virtual void decode_forward(Batch& batch, ModelForwardContext& context) = 0;
        virtual void load_weights(const char* model_path) = 0;
        virtual ~IModel() {}

        virtual void stage_prefill_forward(Batch& batch, ModelForwardContext& context) = 0;
        virtual void stage_decode_forward(Batch& batch, ModelForwardContext& context) = 0;
};

class ModelFactory {
    public:
        static std::unique_ptr<IModel> create_model(const std::string& model_name);
};