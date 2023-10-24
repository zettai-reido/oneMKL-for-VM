#pragma once

#include <cstdint>

namespace vecma::detail {

enum class Status : uint32_t {
    kNotDefined = 0x0,
    kSuccess = 0x0,

    kDomainError = 0x1,
    kSingularity = 0x2,
    kUnderflow = 0x4,
    kOverflow = 0x8,

    // all above 0xFF are not computational
    kComputationEmpty = 0x100,
    kParameterError = 0x101,
    kComputationMask = 0x1FF,

    kRuntimeError = 0xE000'0000,
    kRuntimeMask = 0xE000'0000
};

} // namespace vecma::detail

