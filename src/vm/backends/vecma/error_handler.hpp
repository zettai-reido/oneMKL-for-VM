#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

#include "scalar_args.hpp"
#include "strider.hpp"
#include "vector_arg.hpp"
#include "xtypes.hpp"

namespace vecma::detail {

struct ErrorHandler {
    Status to_fix;
    Argument fix1;
    Argument fix2;

    VectorArgument status;
    VectorArgument g_status;

    ErrorHandler(): to_fix(Status::kNotDefined), fix1{}, fix2{}, status{}, g_status{} { /* empty */ }
    ErrorHandler(ErrorHandler const&) = default;

    template <typename TypeOut>
    static ErrorHandler make(Status _to_fix, TypeOut value, VectorArgument _status, VectorArgument _g_status) {
        ErrorHandler eh;

        eh.fix1.set<TypeOut>(value);
        if constexpr (std::is_same_v<TypeOut, __half>
            || std::is_same_v<TypeOut, float>
            || std::is_same_v<TypeOut, double>
        ) {
            double value2 = sqrt(1.0 - value * value);
            eh.fix2.set<TypeOut>(value2);
        }

        eh.to_fix = _to_fix;
        eh.status = _status;
        eh.g_status = _g_status;
        return eh;
    }

    bool is_enabled() const {
        return (status.get_len() > 0) || (g_status.get_len() > 0) || (to_fix != Status::kNotDefined);
    }
};

} // namespace vecma::detail

