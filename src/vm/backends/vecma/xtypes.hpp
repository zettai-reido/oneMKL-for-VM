#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <type_traits>

#include "mode.hpp"
#include "status.hpp"

#ifdef __CUDACC__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_DEVICE_HOST __device__ __host__
#else
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_DEVICE_HOST
#endif


namespace vecma::detail {

static constexpr int kMaxI = 4;
static constexpr int kMaxO = 2;
static constexpr int kMaxC = 4;


struct Slice {
    size_t start;
    size_t size;
    int64_t stride;
    bool empty;

    explicit constexpr Slice(size_t _start, size_t _size, int64_t _stride): start(_start), size(_size), stride(_stride), empty (0 == _size) { }
    constexpr Slice(size_t n = 0): start(0), size(n), stride(+1), empty (0 == n) { }
};

template <typename xT>
struct BitOperations { static constexpr const bool enabled = false; };

template<> struct BitOperations<Status> { static constexpr const bool enabled = true; };

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, xT>
CUDA_DEVICE_HOST
operator !(xT a) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    return (0 == xa);
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, xT>
CUDA_DEVICE_HOST
operator ~(xT a) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xr = ~xa;
    return static_cast<xT>(xr);
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, xT>
CUDA_DEVICE_HOST
operator &(xT a, xT b) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xr = xa & xb;
    return static_cast<xT>(xr);
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, xT&>
CUDA_DEVICE_HOST
operator &=(xT& a, xT b) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xr = xa & xb;
    a = static_cast<xT>(xr);
    return a;
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, xT>
CUDA_DEVICE_HOST
operator |(xT a, xT b) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xr = xa | xb;
    return static_cast<xT>(xr);
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, xT&>
CUDA_DEVICE_HOST
operator |=(xT& a, xT b) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xr = xa | xb;
    a = static_cast<xT>(xr);
    return a;
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, xT>
CUDA_DEVICE_HOST
operator ^(xT a, xT b) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xr = xa ^ xb;
    return static_cast<xT>(xr);
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, xT&>
CUDA_DEVICE_HOST
operator ^=(xT& a, xT b) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xr = xa ^ xb;
    a = static_cast<xT>(xr);
    return a;
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, bool>
CUDA_DEVICE_HOST
all_of(xT a, xT b) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xr = xa ^ xb;
    return (0 == xr);
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, bool>
CUDA_DEVICE_HOST
any_of(xT a, xT b) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xr = xa & xb;
    return (xr != 0);
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, bool>
CUDA_DEVICE_HOST
none_of(xT a, xT b) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xr = xa & xb;
    return (xr == 0);
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, bool>
CUDA_DEVICE_HOST
all_of_in(xT a, xT b, xT mask) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xmask = static_cast<xU>(mask);
    auto xr = (xa ^ xb) & xmask;
    return (0 == xr);
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, bool>
CUDA_DEVICE_HOST
any_of_in(xT a, xT b, xT mask) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xmask = static_cast<xU>(mask);
    auto xr = (xa & xb) & xmask;
    return (xr != 0);
}

template <typename xT>
std::enable_if_t<BitOperations<xT>::enabled, bool>
CUDA_DEVICE_HOST
none_of_in(xT a, xT b, xT mask) {
    using xU = typename std::underlying_type_t<xT>;
    auto xa = static_cast<xU>(a);
    auto xb = static_cast<xU>(b);
    auto xmask = static_cast<xU>(mask);
    auto xr = (xa & xb) & xmask;
    return (xr == 0);
}


using StatusT = typename std::underlying_type_t<Status>;
using AtomicStatusT = std::atomic<StatusT>;

using StreamIdT = unsigned long long;

} // namespace vecma::detail

