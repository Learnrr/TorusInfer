#include "define.h"
#include "Tensor.h"
#include "Attention.h"
#include "MLP.h"
#include "ModelWeights.h"
#include "Workspace.h"
#include "Layer.h"
#include "ForwardContext.h"

class TransformerLayer: public Layer {
    public:
        TransformerLayer(
            int hidden_size, 
            int num_heads, 
            LayerWeightLayout* layer_layout
        ) {
            attention = std::make_unique<Attention>(hidden_size, num_heads);
            mlp = std::make_unique<MLP>(hidden_size, INTERMEDIATE_SIZE);
            this->layer_layout = layer_layout;
        }

        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;

        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;

    private:
        std::unique_ptr<Attention> attention;
        std::unique_ptr<MLP> mlp;
        LayerWeightLayout* layer_layout;


}