
#pragma once

#include <cstddef>

enum class DataType {
    FLOAT32,
    FLOAT16
};

inline constexpr size_t DataTypeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return 4;
        case DataType::FLOAT16:
            return 2;
    }
    return 0;
}

inline constexpr const char* DataTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return "FLOAT32";
        case DataType::FLOAT16:
            return "FLOAT16";
    }
    return "UNKNOWN";
}

typedef DataType::FLOAT32 float32;
