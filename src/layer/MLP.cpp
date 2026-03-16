#include "MLP.h"

void MLP::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    if (linears.empty()) {
        output = input;
        return;
    }

    Tensor current = input;
    Tensor stage_output = output;

    for (size_t i = 0; i < linears.size(); ++i) {
        linears[i]->prefill_forward(current, stage_output, context);
        if (swiglu && i == mlp_config.activation_after_linear_idx) {
            swiglu->prefill_forward(stage_output, stage_output, context);
        }
        current = stage_output;
    }

    output = current;

}

void MLP::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    if (linears.empty()) {
        output = input;
        return;
    }

    Tensor current = input;
    Tensor stage_output = output;

    for (size_t i = 0; i < linears.size(); ++i) {
        linears[i]->decode_forward(current, stage_output, context);
        if (swiglu && i == mlp_config.activation_after_linear_idx) {
            swiglu->decode_forward(stage_output, stage_output, context);
        }
        current = stage_output;
    }

    output = current;
}