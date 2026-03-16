#include <cuda_runtime.h>
#include <memory>

struct CudaDeleter {
    template <typename T>
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;