#ifndef VECMA_DETAIL_MATH_INLINES_HPP
#define VECMA_DETAIL_MATH_INLINES_HPP 1

#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>

namespace vecma::detail::scalar {

namespace {

template <typename ToType, typename FromType>
inline constexpr ToType bitcast(FromType v) {
    static_assert(sizeof(ToType) == sizeof(FromType), "sizeof(ToType) == sizeof(FromType) is a must");

    uint8_t w[sizeof(ToType)] = {0};

    ToType r;
    memcpy(w, &v, sizeof(w));
    memcpy(&r, w, sizeof(w));
    return r;
}

inline constexpr uint8_t sign(__half v) {
    uint16_t w = bitcast<uint16_t>(v);
    return (w >> 15) & 1;
}

inline constexpr uint8_t expo(__half v) {
    uint16_t w = bitcast<uint16_t>(v);
    return (w >> 10) & 0x1F;
}

inline constexpr uint16_t mant(__half v) {
    uint16_t w = bitcast<uint16_t>(v);
    return w & 0x3FF;
}

inline constexpr uint8_t sign(float v) {
    uint32_t w = bitcast<uint32_t>(v);
    return (w >> 31) & 1;
}

inline constexpr uint8_t expo(float v) {
    uint32_t w = bitcast<uint32_t>(v);
    return (w >> 23) & 0xFF;
}

inline constexpr uint32_t mant(float v) {
    uint32_t w = bitcast<uint32_t>(v);
    return w & 0x7FFFFF;
}

inline constexpr uint8_t sign(double v) {
    uint64_t w = bitcast<uint64_t>(v);
    return (w >> 63) & 1;
}

inline constexpr uint16_t expo(double v) {
    uint64_t w = bitcast<uint64_t>(v);
    return (w >> 52) & 0x7FF;
}

inline constexpr uint64_t mant(double v) {
    uint64_t w = bitcast<uint64_t>(v);
    return w & UINT64_C(0xF'FFFF'FFFF'FFFFF);
}

inline constexpr bool is_inf(__half v) { return (expo(v) == 0x1F && mant(v) == 0); }
inline constexpr bool is_nan(__half v) { return (expo(v) == 0x1F && mant(v) != 0); }
inline constexpr bool is_zero(__half v) { return (expo(v) == 0 && mant(v) == 0); }
inline constexpr bool is_denorm(__half v) { return false; }
inline constexpr bool is_absolute_one(__half v) { return (expo(v) == 0x0F && mant(v) == 0); }
inline constexpr bool is_over_absolute_one(__half v) { return (expo(v) > 0x0F || (expo(v) == 0x0F && mant(v) > 0)); }

inline constexpr bool is_inf(float v) { return (expo(v) == 0xFF && mant(v) == 0); }
inline constexpr bool is_nan(float v) { return (expo(v) == 0xFF && mant(v) == 0); }
inline constexpr bool is_zero(float v) { return (expo(v) == 0 && mant(v) == 0); }
inline constexpr bool is_denorm(float v) { return (expo(v) == 0 && mant(v) != 0); }
inline constexpr bool is_absolute_one(float v) { return (expo(v)  == 0x7F && mant(v) == 0); }
inline constexpr bool is_over_absolute_one(float v) { return (expo(v) > 0x7F || (expo(v) == 0x7F && mant(v) > 0)); }

inline constexpr bool is_inf(double v) { return (expo(v) == 0x7FF && mant(v) == 0); }
inline constexpr bool is_nan(double v) { return (expo(v) == 0x7FF && mant(v) != 0); }
inline constexpr bool is_zero(double v) { return (expo(v) == 0 && mant(v) == 0); }
inline constexpr bool is_denorm(double v) { return (expo(v) == 0 && mant(v) != 0); }
inline constexpr bool is_absolute_one(double v) { return (0x3FF == expo(v) && 0 == mant(v)); }
inline constexpr bool over_absolute_one(double v) { return (expo(v) > 0x3FF || (expo(v) == 0x3FF && mant(v) > 0)); }

} // anon.

} // namespace vecma::detail::scalar

#endif

