#pragma once
#include "Linear.h"
#include "SwiGLU.h"
#include "Tensor.h"
#include "Multiply.h"
#include "Layer.h"
#include "ModelWeights.h"
#include "ForwardContext.h"
class MLP: public Layer {
    public:
        MLP(int hidden_size, int intermediate_size, LayerWeightLayout* layer_layout): layer_layout(layer_layout){
            linear1 = std::make_unique<Linear>(hidden_size, intermediate_size);
            linear2 = std::make_unique<Linear>(intermediate_size, hidden_size);
            swiglu = std::make_unique<SwiGLU>(intermediate_size);
        }
        void forward(Tensor& input, Tensor& output, ForwardContext& context) override;
    private:

        std::unique_ptr<Linear> linear1;
        std::unique_ptr<Linear> linear2;
        std::unique_ptr<Linear> linear3;

        std::unique_ptr<SwiGLU> swiglu;

        LayerWeightLayout* layer_layout;

};