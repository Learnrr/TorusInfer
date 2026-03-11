#pragma once

#include "define.h"
#include "embedding.h"
#include "ModelConfig.h"
#include "ModelWeights.h"
#include "Batch.h"
#include "Workspace.h"
#include "Layer.h"
#include "Tensor.h"
#include "TransformerLayer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "ForwardContext.h"
class Model {
    public:
        Model(const char* model_path) {     
            init(model_path);
        } 

        void init(const char* model_path);

        void prefill_forward(Batch& batch, Workspace& workspace);

        void decode_forward(Batch& batch, Workspace& workspace);

    private:
     
        std::unique_ptr<Embedding> embedding;
        std::unique_ptr<ModelWeights> weights;
        ModelConfig config;
        std::unique_ptr<TransformerLayer[]> transformer_layers; // Array of layers (e.g., attention, MLP, etc.)
        std::unique_ptr<LayerNorm> layer_norm;
        std::unique_ptr<Linear> lm_head;
}