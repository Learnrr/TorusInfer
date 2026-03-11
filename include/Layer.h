#pragma once
#include "Tensor.h"
#include "ForwardContext.h"
class Layer {
    public:
        virtual void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) = 0;
        virtual void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) = 0;
        virtual ~Layer() {} // Virtual destructor

};