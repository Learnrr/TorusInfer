#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include "define.h"
class Tensor {
public:
    void* data;
    size_t size; // Byte size.
    DataType dtype;
    std::vector<size_t> shape;
    std::string device;

    Tensor()
        : data(nullptr), size(0), dtype(DataType::FLOAT16), shape(), device("gpu") {}

    Tensor(size_t num_elements, void* data_ptr, std::vector<size_t> shape, DataType dtype, std::string device = "gpu")
        : data(data_ptr),
          size(num_elements * element_size_bytes(dtype)),
          dtype(dtype),
          shape(std::move(shape)),
          device(std::move(device)) {}

    ~Tensor() {}

    Tensor(const Tensor& other)
        : data(nullptr), size(other.size), dtype(other.dtype), shape(other.shape), device(other.device) {
        if (other.data != nullptr && size > 0) {
            data = new char[size];
            std::memcpy(data, other.data, size);
        }
    }

    static size_t element_size_bytes(DataType dtype) {
        return dtype == DataType::FLOAT16 ? 2 : 4;
    }

    size_t numel() const {
        const size_t elem_bytes = element_size_bytes(dtype);
        return elem_bytes == 0 ? 0 : (size / elem_bytes);
    }

    void view(std::vector<size_t> new_shape) {
        if (data != nullptr) {
            shape = std::move(new_shape);
        }
    }

    Tensor transpose() {
        if (shape.size() != 2) {
            return Tensor(numel(), nullptr, shape, dtype, device);
        }

        Tensor out(numel(), nullptr, {shape[1], shape[0]}, dtype, device);
        out.data = new char[out.size];
        if (data == nullptr) {
            return out;
        }

        if (dtype == DataType::FLOAT32) {
            const float* in = static_cast<const float*>(data);
            float* dst = static_cast<float*>(out.data);
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    dst[j * shape[0] + i] = in[i * shape[1] + j];
                }
            }
        } else {
            const uint16_t* in = static_cast<const uint16_t*>(data);
            uint16_t* dst = static_cast<uint16_t*>(out.data);
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    dst[j * shape[0] + i] = in[i * shape[1] + j];
                }
            }
        }
        return out;
    }

    bool operator==(const Tensor& other) const {
        if (size != other.size || shape != other.shape || dtype != other.dtype) {
            return false;
        }
        if (data == nullptr || other.data == nullptr) {
            return data == other.data;
        }
        return std::memcmp(data, other.data, size) == 0;
    }
};