#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

#include "scalar_args.hpp"
#include "strider.hpp"
#include "vector_arg.hpp"
#include "xtypes.hpp"


namespace vecma::detail {

struct VectorArgs {
    VectorArgument ia[kMaxI];
    VectorArgument oa[kMaxO];
    Argument ca[kMaxC];

    VectorArgument status;

    template <typename TypeIn, typename TypeC, int ni, int nc, bool cyclic = false>
    constexpr ScalarArgs gather(Strider const& str, size_t i) {
        ScalarArgs sa;

        if constexpr (ni > 0) {
            TypeIn a = ia[0].get_c<TypeIn>()[str.index_in<0, cyclic>(i)];
            sa.set_i<TypeIn, 0>(a);
        }
        if constexpr (ni > 1) {
            TypeIn a = ia[1].get_c<TypeIn>()[str.index_in<1, cyclic>(i)];
            sa.set_i<TypeIn, 1>(a);
        }
        if constexpr (ni > 2) {
            TypeIn a = ia[2].get_c<TypeIn>()[str.index_in<2, cyclic>(i)];
            sa.set_i<TypeIn, 2>(a);
        }
        if constexpr (ni > 3) {
            TypeIn a = ia[3].get_c<TypeIn>()[str.index_in<3, cyclic>(i)];
            sa.set_i<TypeIn, 3>(a);
        }

        if constexpr (nc > 0) {
            TypeC v = ca[0].get<TypeC>();
            sa.set_c<TypeC, 0>(v);
        }
        if constexpr (nc > 1) {
            TypeC v = ca[1].get<TypeC>();
            sa.set_c<TypeC, 1>(v);
        }
        if constexpr (nc > 2) {
            TypeC v = ia[2].get<TypeC>();
            sa.set_c<TypeC, 2>(v);
        }
        if constexpr (nc > 3) {
            TypeC v = ia[3].get<TypeC>();
            sa.set_c<TypeC, 3>(v);
        }

        return sa;
    }

    template <typename TypeOut, int no, bool has_error_handler>
    constexpr void scatter(ScalarArgs& sa, Strider const& str, size_t i) {
        if constexpr (no > 0) {
            TypeOut v = sa.get_o<TypeOut, 0>();
            oa[0].template get<TypeOut>()[str.index_out<0>(i)] = v;
            if constexpr (has_error_handler) {
                if (sa.st != Status::kNotDefined) {
                    status.template get<Status>()[str.index_out<0>(i)] = sa.st;
                }
            }
        }
        if constexpr (no > 1) {
            TypeOut v = sa.get_o<TypeOut, 1>();
            oa[1].get<TypeOut>()[str.index_out<1>(i)] = v;
        }
    }
}; // VectorArgs

} // namespace vecma::detail

