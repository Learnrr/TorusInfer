
#include "Layer.h"
#include "Tensor.h"
#include "ForwardContext.h"
#include "ModelConfig.h"
class LayerNorm: public Layer {
    public:
        LayerNorm(std::shared_ptr<LayerNormLayerConfig> config){
            this->config = config;
        }
        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
    private:
        std::shared_ptr<LayerNormLayerConfig> config;

};