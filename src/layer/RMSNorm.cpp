#include "layer/RMSNorm.h"
#include "kernel/rmsnorm_kernel.h"

void RMSNorm::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) {

	size_t hidden_size = config.norm_size;
	if (hidden_size == 0 && !input.shape.empty()) {
		hidden_size = input.shape.back();
	}

	if (hidden_size == 0 
		|| input.data == nullptr 
		|| output.data == nullptr) {
		return;
	}

	size_t num_tokens = context.batch->num_tokens;

	const void* gamma_ptr = (gamma != nullptr ? gamma : norm_weight.data);

	launch_rmsnorm_kernel(
		input.data,
		gamma_ptr,
		output.data,
		num_tokens,
		hidden_size,
		kDefaultEps,
		input.dtype
	);
}

void RMSNorm::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
	size_t hidden_size = config.norm_size;
	if (hidden_size == 0 && !input.shape.empty()) {
		hidden_size = input.shape.back();
	}

	if (hidden_size == 0 
		|| input.data == nullptr 
		|| output.data == nullptr) {
		return;
	}

	size_t num_tokens = context.batch->num_tokens;

	const void* gamma_ptr = (gamma != nullptr ? gamma : norm_weight.data);
	

	launch_rmsnorm_kernel(
		input.data,
		gamma_ptr,
		output.data,
		num_tokens,
		hidden_size,
		kDefaultEps,
		input.dtype
	);
}
