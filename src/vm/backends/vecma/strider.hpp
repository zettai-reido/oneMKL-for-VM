#pragma once

#include <array>
#include <cstdint>
#include <cstddef>

#include "xtypes.hpp"


namespace vecma::detail {

static constexpr size_t kMaxSize = UINT64_C(0xEEEE'EEEE'EEEE'EEEE);

enum class StriderError : int {
    kNormal = 0,
    kNoOutput = 1,
    kSliceEmpty = 2,
    k64BitOverflow = 3,
    kNotEnoughData = 4
};


class Strider {
    size_t m_n_eval;
    Slice m_input[kMaxI];
    Slice m_output[kMaxO];

    size_t m_pp[kMaxI];

    bool m_cyclic;
public:
    constexpr Strider(): m_n_eval(0), m_input{}, m_output{}, m_pp{1}, m_cyclic(false) { /* empty */ }
    constexpr Strider(Strider const&) = default;

    CUDA_HOST
    constexpr void set_cyclic() { m_cyclic = true; }

    CUDA_HOST
    constexpr bool is_cyclic() { return m_cyclic; }

    CUDA_HOST
    constexpr void set_input(int i, Slice sl) { m_input[i] = sl; }

    CUDA_HOST
    constexpr void set_output(int i, Slice sl) { m_output[i] = sl; }

    template <int n_i, int n_o>
    CUDA_HOST
    constexpr StriderError precompute() {
        if constexpr(0 == n_i) { return StriderError::kNormal; }
        if constexpr(0 == n_o) { return StriderError::kNoOutput; }

        size_t p = 1;
        for (int i = n_i - 1; i >= 0; --i) {
            if (m_input[i].empty) { return StriderError::kSliceEmpty; }

            m_pp[i] = p;
            size_t new_p = p * m_input[i].size;
            if (new_p < p) { return StriderError::k64BitOverflow; }
            p = new_p;
        }

        size_t n = kMaxSize;
        for (int i = 0; i < n_o; ++i) {
            if (m_output[i].empty) { return StriderError::kNoOutput; }
            if (m_output[i].size < n) { n = m_output[i].size; }
        }

        if (0 == n) { return StriderError::kNoOutput; }

        for (int i = 0; i < n_i; ++i) {
            if (m_input[i].size < n && !m_cyclic) { return StriderError::kNotEnoughData; }
        }
        m_n_eval = n;
        return StriderError::kNormal;
    }

    template <int i, bool cyclic>
    CUDA_DEVICE
    constexpr size_t index_in(size_t index) const {
        if constexpr(!cyclic) { return m_input[i].start + m_input[i].stride * index; }
        size_t r = index;
        size_t q = index;
        for (int k = 0; k < i; ++k) {
            q = r / m_pp[k];
            r -= m_pp[k] * q;
        }
        return (q % m_input[i].size);
    }

    template <int i>
    CUDA_DEVICE
    constexpr size_t index_out(size_t index) const { return m_output[i].start + m_output[i].stride * index; }

    CUDA_DEVICE_HOST
    constexpr size_t get_neval() const { return m_n_eval; }
}; // class Strider

} // namespace vecma::detail

