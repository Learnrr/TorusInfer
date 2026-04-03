#pragma once

#include "define.h"
#include "layer/Embedding.h"
#include "llm_engine_config.h"
#include "ModelWeights.h"
#include "Batch.h"
#include "Workspace.h"
#include "layer/Layer.h"
#include "Tensor.h"
#include "TransformerLayer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "ForwardContext.h"
#include "IModel.h"
#include "PostProcessor.h"
#include <vector>
class QWEN_Model : public IModel {
    public:
        QWEN_Model() {} 

        void init(LLMEngineConfig& config) override;

        void prefill_forward(Batch& batch, Workspace& workspace, SequencePool* seq_pool = nullptr) override;

        void decode_forward(Batch& batch, Workspace& workspace, SequencePool* seq_pool = nullptr) override;

        void load_weights(const char* model_path) override;

        void stage_prefill_forward(
            Batch& batch,
            Workspace& workspace,
            size_t start_layer,
            size_t end_layer,
            void* external_hidden_in = nullptr,
            SequencePool* seq_pool = nullptr
        ) override;
        void stage_decode_forward(
            Batch& batch,
            Workspace& workspace,
            size_t start_layer,
            size_t end_layer,
            void* external_hidden_in = nullptr,
            SequencePool* seq_pool = nullptr
        ) override;

    private:
        size_t stage_start_layer = 0;
        size_t stage_end_layer = 0;
        bool is_first_stage = true;
        bool is_last_stage = true;
     
        std::unique_ptr<Embedding> embedding;
        std::unique_ptr<ModelWeights> weights;
        LLMEngineConfig config;
        std::vector<std::unique_ptr<Layer>> layers;
        std::unique_ptr<PostProcessor> post_processor;
    };