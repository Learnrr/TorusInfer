#pragma once
#include "define.h"
#include "Tensor.h"
#include "Workspace.h"
#include "Layer.h"
#include "ForwardContext.h"
#include "ModelWeights.h"
#include "ModelConfig.h"
class Linear: public Layer {
    public:
        Linear(const LinearConfig& config, 
            size_t layer_id, 
            Tensor& linear_weight):
        ){
            this->layer_id = layer_id;
            this->linear_weight = linear_weight;
            this->config = config;
        }
        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
    private:
        void run_linear(const Tensor& input, Tensor& output, ForwardContext& context);
        size_t layer_id;
        Tensor& linear_weight;
        LinearConfig config;
};