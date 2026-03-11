#pragma once
#include "define.h"
#include "Tensor.h"
#include "Workspace.h"
#include "Layer.h"
#include "ForwardContext.h"
class Linear: public Layer {
    public:
        Linear(int input_size, int output_size){
            this->input_size = input_size;
            this->output_size = output_size;
        }
        void forward(Tensor& input, Tensor& output, ForwardContext& context) override {
            // Implement the logic for the forward pass of the linear layer


        }
    private:
        size_t input_size;
        size_t output_size;

};