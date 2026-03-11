#include "MLP.h"
void MLP::forward(Tensor& input, Tensor& output, ForwardContext& context) {

    Tensor intermediate_output(intermediate_size, {input.shape[0], intermediate_size}, input.dtype);

    linear1->forward(input, intermediate_output, context);
    swiglu->forward(intermediate_output, intermediate_output, context);
    linear2->forward(intermediate_output, output, context);
    
}