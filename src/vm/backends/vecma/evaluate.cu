#include "device_error_handler.hpp"
#include "scalar_args.hpp"
#include "strider.hpp"
#include "vector_args.hpp"
#include "xtypes.hpp"

#include "scalar.hpp"

namespace vecma::detail {


template <class Functor, bool is_cyclic, bool has_error_handler>
__global__
void kernel(VectorArgs va, Strider str, Functor const& fop = {}, DeviceErrorHandler const& deh = {}) {
    const size_t index = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    if (index >= str.get_neval()) { return; }

    using TypeIn = typename Functor::TypeIn;
    using TypeC = typename Functor::TypeC;
    using TypeOut = typename Functor::TypeOut;
    constexpr int n_i = Functor::n_i;
    constexpr int n_c = Functor::n_c;
    constexpr int n_o = Functor::n_o; 

    (void)n_c;

    ScalarArgs sa;
    sa = va.template gather<TypeIn, TypeC, n_i, n_c, is_cyclic>(str, index);

    Status e_status = fop(sa);
    if constexpr (has_error_handler) {
        Status st = fop.math_check(e_status, sa);
        if (st != Status::kSuccess) { 
            deh.template action<TypeOut, n_o>(sa, st);
            sa.st = st;
        }
        va.template scatter<TypeOut, n_o, true>(sa, str, index);
    } else {
        va.template scatter<TypeOut, n_o, false>(sa, str, index);
    } // if constexpr(has_error_handler)
}

template <class Functor>
Status evaluate(VectorArgs& va, Strider str, ErrorHandler const& eh) {
    constexpr int n_i = Functor::n_i;
    constexpr int n_o = Functor::n_o; 

    using TypeOut = typename Functor::TypeOut;


    str.template precompute<n_i, n_o>();

    size_t neval = str.get_neval();
    int bs = 256;
    int nb = (neval + bs - 1) / bs;

    if (str.is_cyclic()) {
        if (eh.is_enabled()) {
            DeviceErrorHandler deh (eh);
            kernel<Functor, true, true> <<<nb, bs>>>(va, str, Functor(), deh); 
        } else {
            DeviceErrorHandler deh;
            kernel<Functor, true, false> <<<nb, bs>>>(va, str, Functor(), deh); 
        }
    } else {
        kernel<Functor, false, false> <<<nb, bs>>>(va, str, Functor());
    }
    return Status::kSuccess;
};

Status evaluate_pow_h(VectorArgs& va, Strider str, ErrorHandler const& eh) { return evaluate<scalar::PowH>(va, str, eh); }
Status evaluate_pow_s(VectorArgs& va, Strider str, ErrorHandler const& eh) { return evaluate<scalar::PowS>(va, str, eh); }
Status evaluate_pow_d(VectorArgs& va, Strider str, ErrorHandler const& eh) { return evaluate<scalar::PowD>(va, str, eh); }
Status evaluate_pow_c(VectorArgs& va, Strider str, ErrorHandler const& eh) { return evaluate<scalar::PowC>(va, str, eh); }
Status evaluate_pow_z(VectorArgs& va, Strider str, ErrorHandler const& eh) { return evaluate<scalar::PowZ>(va, str, eh); }

} // namespace vecma::detail

