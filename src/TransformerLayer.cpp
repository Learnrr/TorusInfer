#include "TransformerLayer.h"

void TransformerLayer::prefill_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) override {
    // Implement the logic for the prefill forward pass of the transformer layer
    Tensor attn_output(input.size, nullptr input.shape, input.dtype);
    attn_output.data = context.workspace->get_attn_output_workspace();

    attention->prefill_forward(input, attn_output, context);
    mlp->prefill_forward(attn_output, output);
}

void TransformerLayer::decode_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) override {
    // Implement the logic for the decode forward pass of the transformer layer
    Tensor attn_output(input.size, nullptr input.shape, input.dtype);
    attn_output.data = context.workspace->get_attn_output_workspace();

    attention->decode_forward(input, attn_output, context);
    mlp->decode_forward(attn_output, output);
}