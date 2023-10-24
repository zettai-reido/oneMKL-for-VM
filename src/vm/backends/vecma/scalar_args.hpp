#pragma once

#include <complex>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>

#include "xtypes.hpp"

namespace vecma::detail {

union Argument {
    __half m_h;
    float m_s;
    double m_d;
    std::complex<float> m_c;
    std::complex<double> m_z;
    int32_t m_op32;
    int64_t m_op64;

    explicit constexpr Argument(): m_z{0} { memset(this, 0, sizeof(*this)); }
    explicit constexpr Argument(__half h): m_h(h) {}
    explicit constexpr Argument(float s): m_s(s) {}
    explicit constexpr Argument(double d): m_d(d) {}
    explicit constexpr Argument(std::complex<float> c): m_c(c) {}
    explicit constexpr Argument(std::complex<double> z): m_z(z) {}
    explicit constexpr Argument(int32_t op32): m_op32(op32) {}
    explicit constexpr Argument(int64_t op64): m_op64(op64) {}

    template <typename Number> constexpr void set(Number v) {
        if constexpr(std::is_same_v<__half, Number>) { m_h = v; }
        else if constexpr(std::is_same_v<float, Number>) { m_s = v; }
        else if constexpr(std::is_same_v<double, Number>) { m_d = v; }
        else if constexpr(std::is_same_v<std::complex<float>, Number>) { m_c = v; }
        else if constexpr(std::is_same_v<std::complex<double>, Number>) { m_z = v; }
        else if constexpr(std::is_same_v<int32_t, Number>) { m_op32 = v; }
        else if constexpr(std::is_same_v<int64_t, Number>) { m_op64 = v; }
        else if constexpr(std::is_same_v<void, Number>) {}
    }

    template <typename Number> constexpr Number get() const {
        if constexpr(std::is_same_v<__half, Number>) { return m_h; }
        else if constexpr(std::is_same_v<float, Number>) { return m_s; }
        else if constexpr(std::is_same_v<double, Number>) { return m_d; }
        else if constexpr(std::is_same_v<std::complex<float>, Number>) { return m_c; }
        else if constexpr(std::is_same_v<std::complex<double>, Number>) { return m_z; }
        else if constexpr(std::is_same_v<int32_t, Number>) { return m_op32; }
        else if constexpr(std::is_same_v<int64_t, Number>) { return m_op64; }
    }
};

struct ScalarArgs {
    Argument ia[kMaxI];
    Argument oa[kMaxO];
    Argument ca[kMaxC];

    template <typename Number, int k> constexpr Number get_i() const { return ia[k].get<Number>(); }
    template <typename Number, int k> constexpr Number get_o() const { return oa[k].get<Number>(); }
    template <typename Number, int k> constexpr Number get_c() const { return ca[k].get<Number>(); }

    template <typename Number, int k> constexpr Number set_i(Number v) { ia[k].set<Number>(v); }
    template <typename Number, int k> constexpr Number set_o(Number v) { oa[k].set<Number>(v); }
    template <typename Number, int k> constexpr Number set_c(Number v) { ca[k].set<Number>(v); }

    Status st;
};

} // namespace vecma::detail

