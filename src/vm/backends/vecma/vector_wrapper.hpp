#pragma once

#include <memory>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>



namespace vecma::detail {

template <typename Number>
class VectorWrapper {
    bool m_valid;
    bool m_copied;

    Number* m_ptr;
    std::shared_ptr<Number> m_device_ptr;
    bool m_copy_back;
    size_t m_size;
    std::string m_last_cuda_error;

    bool check_cuda_driver_error(CUresult cu) {
        if (CUDA_SUCCESS == cu) { return true; }

        const char* desc = nullptr;
        cuGetErrorString(cu, &desc);
        if (desc != nullptr) { m_last_cuda_error = desc; }
        return false;
    }

    bool check_cuda_runtime_error(cudaError_t e) {
        if (cudaSuccess == e) { return true; }

        const char* desc = nullptr;
        desc = cudaGetErrorString(e);
        if (nullptr != desc) { m_last_cuda_error = desc; }
        return false;
    }

    bool is_device_ptr(const Number* ptr) {
        if (nullptr == ptr) { return true; } // NULLs are universal

        cudaPointerAttributes cpa{};
        cudaError_t r = cudaPointerGetAttributes(&cpa, ptr);
        if (!check_cuda_runtime_error(r)) { return false; }
        return (nullptr != cpa.devicePointer);
    }

    Number* copy_to_device(size_t size, const Number* ptr) {
        Number* dptr;
        size_t bs = size * sizeof(Number);

        cudaError_t e = cudaSuccess;
        e = ::cudaMalloc(&dptr, bs);
        if (!check_cuda_runtime_error(e)) { return nullptr; }
        e = ::cudaMemcpy(dptr, ptr, bs, cudaMemcpyHostToDevice);
        if (!check_cuda_runtime_error(e)) { return nullptr; }

        return dptr;
    }

    void copy_to_heap() {
        if (0 == m_size || nullptr == m_ptr || nullptr == m_device_ptr || m_copied) { return; }
        size_t bs = m_size * sizeof(Number);
        fprintf(stderr, "copy_to_heap: %zu\n", bs);

        cudaError_t e = cudaSuccess;
        e = ::cudaMemcpy(m_ptr, m_device_ptr.get(), bs, cudaMemcpyDeviceToHost);
        if (!check_cuda_runtime_error(e)) { return; }
        m_copied = true;
    }

public:
    bool is_valid() { return m_valid; }
    bool is_copied() { return m_copy_back && m_copied; }

    explicit VectorWrapper(size_t size, Number* ptr): m_copy_back(true), m_valid(true), m_copied(false) {
        m_ptr = ptr;
        if (is_device_ptr(ptr)) { return; }

        auto Deleter = [=](void* ptr) { cudaFree(ptr); };
        m_device_ptr = std::shared_ptr<Number>(copy_to_device(size, m_ptr), Deleter);
        m_size = size;
        m_valid = (nullptr != m_device_ptr);
    }

    explicit VectorWrapper(size_t size, const Number* cptr):
        VectorWrapper(size, const_cast<Number*>(cptr)) { m_copy_back = false; }

    void copy_back() { if (m_copy_back) { copy_to_heap(); } }

    Number* operator()() { return m_device_ptr.get(); }
    inline bool needs_sync() const { return m_copy_back; }

    std::string const& get_last_cuda_error() { return m_last_cuda_error; }

    VectorWrapper(const VectorWrapper&) = default;
    VectorWrapper& operator=(const VectorWrapper&) = default;
    ~VectorWrapper() = default;
};

} // namespace vecma::detail

