#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

#include "scalar_args.hpp"
#include "strider.hpp"
#include "xtypes.hpp"

namespace vecma::detail {

struct VectorArgument {
    union {
        const __half* c_h;
        const float* c_s;
        const double* c_d;
        const std::complex<float>* c_c;
        const std::complex<double>* c_z;

        const int32_t* c_op32;
        const int64_t* c_op64;

        const Status* c_status;
        const void *c_v;

        __half* h;
        float* s;
        double* d;
        std::complex<float>* c;
        std::complex<double>* z;

        int32_t* op32;
        int64_t* op64;

        Status* status;

        void *v;
    } m_pointer;

    size_t m_len;

    explicit constexpr VectorArgument(): m_pointer { .v = nullptr }, m_len(0) { /* empty */ }

    template <typename Number>
    constexpr void set_c(const Number* ptr, size_t len) { 
             if constexpr(std::is_same_v<__half, Number>) { m_pointer.c_h = ptr; m_len = len; }
        else if constexpr(std::is_same_v<float, Number>) { m_pointer.c_s = ptr; m_len = len; }
        else if constexpr(std::is_same_v<double, Number>) { m_pointer.c_d = ptr; m_len = len; }
        else if constexpr(std::is_same_v<std::complex<float>, Number>) { m_pointer.c_c = ptr; m_len = len; }
        else if constexpr(std::is_same_v<std::complex<double>, Number>) { m_pointer.c_z = ptr; m_len = len; }
        else if constexpr(std::is_same_v<int32_t, Number>) { m_pointer.c_op32 = ptr; m_len = len; }
        else if constexpr(std::is_same_v<int64_t, Number>) { m_pointer.c_op64 = ptr; m_len = len; }
        else if constexpr(std::is_same_v<Status, Number>) { m_pointer.c_status = ptr; m_len = len; }
        else if constexpr(std::is_same_v<void, Number>) { m_pointer.c_v = ptr; m_len = len; }
    }

    template <typename Number>
    constexpr void set(Number* ptr, size_t len) { 
             if constexpr(std::is_same_v<__half, Number>) { m_pointer.h = ptr; m_len = len; }
        else if constexpr(std::is_same_v<float, Number>) { m_pointer.s = ptr; m_len = len; }
        else if constexpr(std::is_same_v<double, Number>) { m_pointer.d = ptr; m_len = len; }
        else if constexpr(std::is_same_v<std::complex<float>, Number>) { m_pointer.c = ptr; m_len = len; }
        else if constexpr(std::is_same_v<std::complex<double>, Number>) { m_pointer.z = ptr; m_len = len; }
        else if constexpr(std::is_same_v<int32_t, Number>) { m_pointer.op32 = ptr; m_len = len; }
        else if constexpr(std::is_same_v<int64_t, Number>) { m_pointer.op64 = ptr; m_len = len; }
        else if constexpr(std::is_same_v<Status, Number>) { m_pointer.status = ptr; m_len = len; }
        else if constexpr(std::is_same_v<void, Number>) { m_pointer.v = ptr; m_len = len; }
    }

    template <typename Number> 
    constexpr const Number* get_c() const {
             if constexpr(std::is_same_v<__half, Number>) { return m_pointer.c_h; }
        else if constexpr(std::is_same_v<float, Number>) { return m_pointer.c_s; }
        else if constexpr(std::is_same_v<double, Number>) { return m_pointer.c_d; }
        else if constexpr(std::is_same_v<std::complex<float>, Number>) { return m_pointer.c_c; }
        else if constexpr(std::is_same_v<std::complex<double>, Number>) { return m_pointer.c_z; }
        else if constexpr(std::is_same_v<int32_t, Number>) { return m_pointer.c_op32; }
        else if constexpr(std::is_same_v<int64_t, Number>) { return m_pointer.c_op64; }
        else if constexpr(std::is_same_v<Status, Number>) { return m_pointer.c_status; }
        else if constexpr(std::is_same_v<void, Number>) { return m_pointer.c_v; }
    }

    template <typename Number> 
    constexpr Number* get() const {
        if constexpr(std::is_same_v<__half, Number>) { return m_pointer.h; }
        else if constexpr(std::is_same_v<float, Number>) { return m_pointer.s; }
        else if constexpr(std::is_same_v<double, Number>) { return m_pointer.d; }
        else if constexpr(std::is_same_v<std::complex<float>, Number>) { return m_pointer.c; }
        else if constexpr(std::is_same_v<std::complex<double>, Number>) { return m_pointer.z; }
        else if constexpr(std::is_same_v<int32_t, Number>) { return m_pointer.op32; }
        else if constexpr(std::is_same_v<int64_t, Number>) { return m_pointer.op64; }
        else if constexpr(std::is_same_v<Status, Number>) { return m_pointer.status; }
        else if constexpr(std::is_same_v<void, Number>) { return m_pointer.v; }
    }

    CUDA_DEVICE_HOST
    size_t get_len() const { return m_len; }
}; // VectorArgument

} // namespace vecma::detail

