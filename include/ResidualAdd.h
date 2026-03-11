
#include "Tensor.h"
#include "Layer.h"
class ResidualAdd: public Layer {
    public:
        ResidualAdd(int hidden_size);
        void forward(Tensor& input, Tensor& output, ForwardContext& context) override;
    private:

};