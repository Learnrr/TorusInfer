
#include "Layer.h"
#include "Tensor.h"
#include "ForwardContext.h"
class LayerNorm: public Layer {
    public:
        LayerNorm(int hidden_size);
        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
    private:

};