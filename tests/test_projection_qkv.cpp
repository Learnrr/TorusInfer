/*
cd tests
nvcc -std=c++17 -O2 -I../include -I../include/kernel \
    test_projection_qkv.cpp ../kernel/projection.cu \
    -o ../build/tests/test_projection_qkv.exe
../build/tests/test_projection_qkv.exe
*/

#include "kernel/projection_kernel.h"
#include "Tensor.h"

#include <cassert>
#include <cmath>
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

void SplitQKVLikeAttention(
    const Tensor& qkv,
    Tensor& q,
    Tensor& k,
    Tensor& v,
    size_t batch_seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim
) {
    const size_t q_total = batch_seq_len * num_heads * head_dim;
    const size_t k_total = batch_seq_len * num_kv_heads * head_dim;
    const size_t v_total = batch_seq_len * num_kv_heads * head_dim;

    if (qkv.dtype == DataType::FLOAT32) {
        float* base = static_cast<float*>(qkv.data);
        q.data = base;
        k.data = base + q_total;
        v.data = base + q_total + k_total;
    } else if (qkv.dtype == DataType::FLOAT16 || qkv.dtype == DataType::BF16) {
        uint16_t* base = static_cast<uint16_t*>(qkv.data);
        q.data = base;
        k.data = base + q_total;
        v.data = base + q_total + k_total;
    } else {
        q.data = nullptr;
        k.data = nullptr;
        v.data = nullptr;
    }

    const size_t elem_bytes = Tensor::element_size_bytes(qkv.dtype);
    q.size = q_total * elem_bytes;
    k.size = k_total * elem_bytes;
    v.size = v_total * elem_bytes;
    q.num_elements = q_total;
    k.num_elements = k_total;
    v.num_elements = v_total;
    q.shape = {batch_seq_len, num_heads, head_dim};
    k.shape = {batch_seq_len, num_kv_heads, head_dim};
    v.shape = {batch_seq_len, num_kv_heads, head_dim};
    q.dtype = qkv.dtype;
    k.dtype = qkv.dtype;
    v.dtype = qkv.dtype;
    q.device = qkv.device;
    k.device = qkv.device;
    v.device = qkv.device;
}

void TestProjectionKernelFloat32LayoutAndValues() {
    // Tiny config that still exercises Q/K/V partitioning.
    const size_t batch_seq_len = 2;
    const size_t num_heads = 2;
    const size_t num_kv_heads = 1;
    const size_t head_dim = 2;

    const size_t in_features = num_heads * head_dim;      // 4
    const size_t q_size = num_heads * head_dim;           // 4
    const size_t k_size = num_kv_heads * head_dim;        // 2
    const size_t v_size = num_kv_heads * head_dim;        // 2
    const size_t out_features = q_size + k_size + v_size; // 8

    // Input [seq, in] row-major.
    const std::vector<float> h_input = {
        // token0
        1.0f, 2.0f, 3.0f, 4.0f,
        // token1
        5.0f, 6.0f, 7.0f, 8.0f
    };

    // Weight [in, out] row-major. Make values unique for easier debugging.
    std::vector<float> h_weight(in_features * out_features, 0.0f);
    for (size_t i = 0; i < in_features; ++i) {
        for (size_t o = 0; o < out_features; ++o) {
            h_weight[i * out_features + o] = static_cast<float>((i + 1) * 10 + (o + 1));
        }
    }

    // CPU reference for current kernel contract:
    // 1) compute y[token, out] = sum_i input[token,i] * weight[i,out]
    // 2) store output as [Q_all_tokens][K_all_tokens][V_all_tokens]
    std::vector<float> h_expected(batch_seq_len * out_features, 0.0f);
    std::vector<float> h_expected_out_major(batch_seq_len * out_features, 0.0f);
    std::vector<float> h_expected_q(batch_seq_len * q_size, 0.0f);
    std::vector<float> h_expected_k(batch_seq_len * k_size, 0.0f);
    std::vector<float> h_expected_v(batch_seq_len * v_size, 0.0f);
    for (size_t t = 0; t < batch_seq_len; ++t) {
        for (size_t o = 0; o < out_features; ++o) {
            float sum = 0.0f;
            float sum_out_major = 0.0f;
            for (size_t i = 0; i < in_features; ++i) {
                sum += h_input[t * in_features + i] * h_weight[i * out_features + o];
                sum_out_major += h_input[t * in_features + i] * h_weight[o * in_features + i];
            }

            if (o < q_size) {
                h_expected_q[t * q_size + o] = sum;
            } else if (o < q_size + k_size) {
                h_expected_k[t * k_size + (o - q_size)] = sum;
            } else {
                h_expected_v[t * v_size + (o - q_size - k_size)] = sum;
            }

            size_t dst_idx = 0;
            if (o < q_size) {
                dst_idx = t * q_size + o;
            } else if (o < q_size + k_size) {
                dst_idx = batch_seq_len * q_size + t * k_size + (o - q_size);
            } else {
                dst_idx = batch_seq_len * (q_size + k_size) + t * v_size + (o - q_size - k_size);
            }
            h_expected[dst_idx] = sum;
            h_expected_out_major[dst_idx] = sum_out_major;
        }
    }

    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_input), h_input.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_weight), h_weight.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_output), h_expected.size() * sizeof(float)));

    CheckCuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemset(d_output, 0, h_expected.size() * sizeof(float)));

    launch_projection_kernel(
        d_input,
        d_weight,
        nullptr,
        d_output,
        batch_seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        DataType::FLOAT32
    );

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_output(h_expected.size(), 0.0f);
    CheckCuda(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_match_in_major = true;
    bool all_match_out_major = true;
    for (size_t i = 0; i < h_expected.size(); ++i) {
        if (!AlmostEqual(h_output[i], h_expected[i])) {
            all_match_in_major = false;
        }
        if (!AlmostEqual(h_output[i], h_expected_out_major[i])) {
            all_match_out_major = false;
        }
    }

    if (!all_match_in_major) {
        std::cerr << "Projection mismatch against in-major reference [in,out].\n";
        std::cerr << "idx0 got=" << h_output[0]
                  << " in-major=" << h_expected[0]
                  << " out-major=" << h_expected_out_major[0] << "\n";
        if (all_match_out_major) {
            std::cerr << "Kernel output matches out-major reference [out,in].\n";
        }
    }
    assert(all_match_in_major);

    // Projection -> split validation with the same pointer slicing as Attention::split_qkv.
    Tensor qkv;
    qkv.data = d_output;
    qkv.num_elements = h_expected.size();
    qkv.size = h_expected.size() * sizeof(float);
    qkv.shape = {batch_seq_len, out_features};
    qkv.dtype = DataType::FLOAT32;
    qkv.device = "gpu";

    Tensor q;
    Tensor k;
    Tensor v;
    SplitQKVLikeAttention(qkv, q, k, v, batch_seq_len, num_heads, num_kv_heads, head_dim);

    std::vector<float> h_q(q.num_elements, 0.0f);
    std::vector<float> h_k(k.num_elements, 0.0f);
    std::vector<float> h_v(v.num_elements, 0.0f);
    CheckCuda(cudaMemcpy(h_q.data(), q.data, q.size, cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(h_k.data(), k.data, k.size, cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(h_v.data(), v.data, v.size, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < h_expected_q.size(); ++i) {
        assert(AlmostEqual(h_q[i], h_expected_q[i]));
    }
    for (size_t i = 0; i < h_expected_k.size(); ++i) {
        assert(AlmostEqual(h_k[i], h_expected_k[i]));
    }
    for (size_t i = 0; i < h_expected_v.size(); ++i) {
        assert(AlmostEqual(h_v[i], h_expected_v[i]));
    }

    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_input);
}

} // namespace

int main() {
    if (!HasCudaDevice()) {
        std::cout << "No CUDA device found, skipping test_projection_qkv\n";
        return 0;
    }

    TestProjectionKernelFloat32LayoutAndValues();

    std::cout << "test_projection_qkv passed\n";
    return 0;
}
