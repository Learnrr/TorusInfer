#pragma once

#include <cstddef>
#include "define.h"


void launch_swiglu_kernel_from_gate_up(
	const void* gate,
	const void* up,
	void* output,
	size_t num_tokens,
	size_t hidden_size,
	DataType dtype
);
