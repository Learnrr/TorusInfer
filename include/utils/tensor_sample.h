#pragma once

#include "Tensor.h"
#include "utils/logger.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include "half_float/half.hpp"

namespace tensor_sample {

inline float bf16_bits_to_float(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float v = 0.0f;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

inline bool read_tensor_element_as_float(const Tensor& t, size_t flat_index, float& out) {
    if (t.data == nullptr || flat_index >= t.num_elements) {
        return false;
    }

    const size_t elem_bytes = Tensor::element_size_bytes(t.dtype);
    if (elem_bytes == 0) {
        return false;
    }

    uint8_t buf[4] = {0, 0, 0, 0};
    const char* src = static_cast<const char*>(t.data) + flat_index * elem_bytes;
    cudaError_t err = cudaMemcpy(buf, src, elem_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return false;
    }

    if (t.dtype == DataType::FLOAT32) {
        std::memcpy(&out, buf, sizeof(float));
        return true;
    }
    if (t.dtype == DataType::FLOAT16) {
        uint16_t bits = 0;
        std::memcpy(&bits, buf, sizeof(uint16_t));
        half_float::half h;
        std::memcpy(&h, &bits, sizeof(uint16_t));
        out = static_cast<float>(h);
        return true;
    }
    if (t.dtype == DataType::BF16) {
        uint16_t bits = 0;
        std::memcpy(&bits, buf, sizeof(uint16_t));
        out = bf16_bits_to_float(bits);
        return true;
    }

    return false;
}

inline void log_tensor_flat_sample(
    const Tensor& t,
    const std::string& tag,
    size_t start_index = 0,
    size_t sample_count = 16
) {
    if (t.data == nullptr || t.num_elements == 0) {
        return;
    }

    if (start_index >= t.num_elements) {
        return;
    }

    const size_t n = std::min(sample_count, t.num_elements - start_index);
    std::ostringstream oss;
    oss << "Tensor sample " << tag
        << " start=" << start_index
        << " count=" << n
        << " data=[";

    for (size_t i = 0; i < n; ++i) {
        float v = 0.0f;
        if (!read_tensor_element_as_float(t, start_index + i, v)) {
            LOG_ERROR("Failed reading tensor flat sample");
            return;
        }
        if (i > 0) {
            oss << ", ";
        }
        oss << v;
    }
    oss << "]";
    LOG_INFO(oss.str());
}

inline void log_tensor_matrix_sample(
    const Tensor& t,
    const std::string& tag,
    size_t start_row = 0,
    size_t start_col = 0,
    size_t sample_rows = 1,
    size_t sample_cols = 16
) {
    if (t.data == nullptr || t.shape.size() < 2) {
        return;
    }

    const size_t rows = t.shape[0];
    const size_t cols = t.shape[1];
    if (rows == 0 || cols == 0 || start_row >= rows || start_col >= cols) {
        return;
    }

    const size_t r_count = std::min(sample_rows, rows - start_row);
    const size_t c_count = std::min(sample_cols, cols - start_col);

    std::ostringstream oss;
    oss << "Tensor sample " << tag
        << " shape=[" << rows << "," << cols << "]"
        << " window=(r=" << start_row << ",c=" << start_col
        << ",rows=" << r_count << ",cols=" << c_count << ")"
        << " data=[";

    for (size_t r = 0; r < r_count; ++r) {
        if (r > 0) {
            oss << ", ";
        }
        oss << "[";
        for (size_t c = 0; c < c_count; ++c) {
            const size_t flat = (start_row + r) * cols + (start_col + c);
            float v = 0.0f;
            if (!read_tensor_element_as_float(t, flat, v)) {
                LOG_ERROR("Failed reading tensor matrix sample");
                return;
            }
            if (c > 0) {
                oss << ", ";
            }
            oss << v;
        }
        oss << "]";
    }

    oss << "]";
    LOG_INFO(oss.str());
}

} // namespace tensor_sample
