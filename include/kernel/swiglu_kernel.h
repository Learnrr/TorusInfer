#pragma once

#include <cstddef>

void launch_swiglu_kernel(
	const float* gate,
	const float* up,
	float* output,
	size_t num_tokens,
	size_t hidden_size
);
