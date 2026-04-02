/*
cd tests
nvcc -std=c++17 -O2 -I../include -I../include/layer -I../include/kernel \
    test_rope.cpp ../src/layer/position/RoPE.cpp ../kernel/rope_kernel.cu \
    -o ../build/tests/test_rope.exe
../build/tests/test_rope.exe
*/

#include "layer/position/RoPE.h"
#include "ForwardContext.h"
#include "Batch.h"
#include "Tensor.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) && (count > 0);
}

void CheckCuda(cudaError_t err) {
    assert(err == cudaSuccess);
}

bool AlmostEqual(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

float InvFreq(size_t pair_idx, size_t head_dim, float rope_theta) {
    return std::pow(rope_theta, -2.0f * static_cast<float>(pair_idx) / static_cast<float>(head_dim));
}

void ApplyRopeCpuInplace(
    std::vector<float>& x,
    const std::vector<size_t>& positions,
    size_t num_tokens,
    size_t num_heads,
    size_t head_dim,
    float rope_theta
) {
    const size_t half_dim = head_dim / 2;
    for (size_t t = 0; t < num_tokens; ++t) {
        for (size_t h = 0; h < num_heads; ++h) {
            const size_t base = (t * num_heads + h) * head_dim;
            for (size_t p = 0; p < half_dim; ++p) {
                const float inv_freq = InvFreq(p, head_dim, rope_theta);
                const float angle = static_cast<float>(positions[t]) * inv_freq;
                const float c = std::cos(angle);
                const float s = std::sin(angle);

                const float x0 = x[base + p];
                const float x1 = x[base + p + half_dim];
                x[base + p] = x0 * c - x1 * s;
                x[base + p + half_dim] = x0 * s + x1 * c;
            }
        }
    }
}

void TestRopeApplyFloat32MatchesCpuReference() {
    const size_t num_tokens = 4;
    const size_t num_q_heads = 2;
    const size_t num_kv_heads = 1;
    const size_t head_dim = 8;
    const float rope_theta = 1000000.0f;

    std::vector<size_t> positions = {0, 1, 2, 3};

    std::vector<float> h_q(num_tokens * num_q_heads * head_dim, 0.0f);
    std::vector<float> h_k(num_tokens * num_kv_heads * head_dim, 0.0f);

    for (size_t i = 0; i < h_q.size(); ++i) {
        h_q[i] = 0.01f * static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < h_k.size(); ++i) {
        h_k[i] = -0.02f * static_cast<float>(i + 1);
    }

    std::vector<float> h_q_expected = h_q;
    std::vector<float> h_k_expected = h_k;
    ApplyRopeCpuInplace(h_q_expected, positions, num_tokens, num_q_heads, head_dim, rope_theta);
    ApplyRopeCpuInplace(h_k_expected, positions, num_tokens, num_kv_heads, head_dim, rope_theta);

    float* d_q = nullptr;
    float* d_k = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_q), h_q.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_k), h_k.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice));

    Batch batch;
    batch.token_positions = positions;
    batch.num_tokens = num_tokens;

    ForwardContext ctx{};
    ctx.layer_id = 0;
    ctx.batch = &batch;
    ctx.workspace = nullptr;
    ctx.config = nullptr;

    Tensor q_tensor;
    q_tensor.data = d_q;
    q_tensor.num_elements = h_q.size();
    q_tensor.size = h_q.size() * sizeof(float);
    q_tensor.shape = {num_tokens, num_q_heads, head_dim};
    q_tensor.dtype = DataType::FLOAT32;
    q_tensor.device = "gpu";

    Tensor k_tensor;
    k_tensor.data = d_k;
    k_tensor.num_elements = h_k.size();
    k_tensor.size = h_k.size() * sizeof(float);
    k_tensor.shape = {num_tokens, num_kv_heads, head_dim};
    k_tensor.dtype = DataType::FLOAT32;
    k_tensor.device = "gpu";

    RoPE rope;
    rope.apply(q_tensor, k_tensor, ctx, num_q_heads, num_kv_heads, head_dim, rope_theta);
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_q_out(h_q.size(), 0.0f);
    std::vector<float> h_k_out(h_k.size(), 0.0f);
    CheckCuda(cudaMemcpy(h_q_out.data(), d_q, h_q_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(h_k_out.data(), d_k, h_k_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < h_q_out.size(); ++i) {
        assert(AlmostEqual(h_q_out[i], h_q_expected[i]));
    }
    for (size_t i = 0; i < h_k_out.size(); ++i) {
        assert(AlmostEqual(h_k_out[i], h_k_expected[i]));
    }

    cudaFree(d_q);
    cudaFree(d_k);
}

}  // namespace

int main() {
    if (!HasCudaDevice()) {
        std::cout << "No CUDA device found, skipping test_rope\n";
        return 0;
    }

    TestRopeApplyFloat32MatchesCpuReference();

    std::cout << "test_rope passed\n";
    return 0;
}
