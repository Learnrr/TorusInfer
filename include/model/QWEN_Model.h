#pragma once

#include "define.h"
#include "layer/Embedding.h"
#include "ModelConfig.h"
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

        void init(ModelConfig config);

        void prefill_forward(Batch& batch, Workspace& workspace);

        void decode_forward(Batch& batch, Workspace& workspace);

        void load_weights(const char* model_path);

    private:
     
        std::unique_ptr<Embedding> embedding;
        std::unique_ptr<ModelWeights> weights;
        ModelConfig config;
        std::vector<std::unique_ptr<Layer>> layers;
        std::unique_ptr<PostProcessor> post_processor;
    };