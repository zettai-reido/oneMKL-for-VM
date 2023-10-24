#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cmath>

#include "error_handler.hpp"
#include "scalar_args.hpp"
#include "vector_arg.hpp"
#include "xtypes.hpp"

namespace vecma::detail {

struct DeviceErrorHandler {
    Status to_fix;
    Argument fix1;
    Argument fix2;

    VectorArgument status;
    VectorArgument g_status;

    constexpr DeviceErrorHandler(): to_fix(Status::kNotDefined), fix1{}, fix2{}, status{}, g_status{} { /* empty */ }
    constexpr DeviceErrorHandler(DeviceErrorHandler const&) = default;
    constexpr DeviceErrorHandler(ErrorHandler const& eh): to_fix(eh.to_fix), fix1(eh.fix1), fix2(eh.fix2), status(eh.status), g_status(eh.g_status) { }

    template <typename TypeOut, int n_o>
    CUDA_DEVICE
    void action(ScalarArgs& sa, Status st) const {
        if (status.get_len() == 1) { 
            StatusT* p = static_cast<StatusT*>(static_cast<void*>(status.get<Status>()));
            StatusT v = static_cast<StatusT>(st);
            atomicOr(p, v);
        }

        if (g_status.get_len() == 1) {
            StatusT* p = static_cast<StatusT*>(static_cast<void*>(status.get<Status>()));
            StatusT v = static_cast<StatusT>(st);
            atomicOr(p, v);
        }

        // value replacement
        if (!any_of(to_fix, st)) { return; }
        if constexpr (n_o > 0) { 
            TypeOut v = fix1.get<TypeOut>();
            sa.set_o<TypeOut, 0>(v);
        }
        if constexpr (n_o > 1) {
            TypeOut v = fix2.get<TypeOut>();
            sa.set_o<TypeOut, 1>(v);
        }
    }
}; // struct DeviceErrorHandler

} // namespace vecma::detail

