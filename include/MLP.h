#pragma once
#include "Linear.h"
#include "SwiGLU.h"
#include "Tensor.h"
#include "Multiply.h"
#include "Layer.h"
#include "ModelWeights.h"
#include "ForwardContext.h"
#include "ModelConfig.h"
#include <vector>
class MLP: public Layer {
    public:
        MLP(const MLPLayerConfig& mlp_config,
            MLPLayerWeightLayout& layer_layout):
             layer_layout(layer_layout), 
             mlp_config(mlp_config){

            linears.resize(mlp_config.mlp_linears.size());
            for(size_t i = 0; i < mlp_config.mlp_linears.size(); ++i){
                Tensor& linear_weight;
                if (i == 0) {
                    linear_weight = layer_layout.gate_proj_weight;
                } else if (i == 1) {
                    linear_weight = layer_layout.up_proj_weight;
                } else {
                    linear_weight = layer_layout.down_proj_weight;
                }
                linears[i] = std::make_unique<Linear>(
                    mlp_config.mlp_linears[i],
                    i+1, 
                    linear_weight
                );
            }
            swiglu = std::make_unique<SwiGLU>(mlp_config.intermediate_size);
        }
        void forward(Tensor& input, Tensor& output, ForwardContext& context) override;
    private:

        std::vector<std::unique_ptr<Linear>> linears;
        std::unique_ptr<SwiGLU> swiglu;

        MLPLayerWeightLayout& layer_layout;
        MLPLayerConfig mlp_config;


};