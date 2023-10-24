#include <limits>
#include <complex>
#include <cuda_fp16.h>

#include "xtypes.hpp"
#include "math_inlines.hpp"
#include "scalar_args.hpp"

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#define __half _Float16
#warning "Compiling for host"
#endif

namespace vecma::detail::scalar {

struct Cdfnorminv {
    static constexpr const int n_i = 1;
    static constexpr const int n_c = 0;
    static constexpr const int n_o = 1;

    CUDA_DEVICE
    Status math_check(Status e_result, ScalarArgs const& sa) const { return e_result; }
};

struct CdfnorminvH {
    static constexpr const int n_i = 1;
    static constexpr const int n_c = 0;
    static constexpr const int n_o = 1;

    using TypeIn = __half;
    using TypeC = void;
    using TypeOut = __half;

    CUDA_DEVICE
    Status operator()(ScalarArgs& sa) const {
        __half s_a = sa.template get_i<__half, 0>();
        __half s_y = __half(0);

//--------------------------------------
        float t1 = s_a;
        float t2 = fmaf(2.0, t1, -1.0);
        float t3 = erfinvf(t2);
        float t4 = 0x1.6A09E667F3BCC908p+1f * t3;
        float iar = t4;
//--------------------------------------

        s_y = static_cast<__half>(iar);

        if (is_inf(s_y) && !is_inf(iar)) { return Status::kOverflow; }
        if (is_zero(s_y) && !is_zero(iar)) { return Status::kUnderflow; }

        sa.set_o<__half, 0>(s_y);
        return Status::kSuccess;
    }

    CUDA_DEVICE
    Status math_check(Status e_result, ScalarArgs const& sa) const { 
        __half s_a = sa.template get_i<__half, 0>();

        if (is_over_absolute_one(s_a)) { return Status::kDomainError; }
        if (is_absolute_one(s_a)) { return Status::kSingularity; }
        return e_result; 
    }
}; // struct CdfnorminvH

struct CdfnorminvS {
    static constexpr const int n_i = 1;
    static constexpr const int n_c = 0;
    static constexpr const int n_o = 1;

    using TypeIn = float;
    using TypeC = void;
    using TypeOut = float;

    CUDA_DEVICE
    Status operator()(ScalarArgs& sa) const {
        float s_a = sa.template get_i<float, 0>();
        float s_y = float(0);

        float t1 = s_a;
        float t2 = fmaf(2.0, t1, -1.0);
        float t3 = erfinvf(t2);
        float t4 = 0x1.6A09E667F3BCC908p+1f * t3;

        s_y = t4;

        sa.set_o<float, 0>(s_y);
        return Status::kSuccess;
    }

    CUDA_DEVICE
    Status math_check(Status e_result, ScalarArgs const& sa) const { 
        float s_a = sa.template get_i<float, 0>();

        if (is_over_absolute_one(s_a)) { return Status::kDomainError; }
        if (is_absolute_one(s_a)) { return Status::kSingularity; }
        return e_result; 
    }
};

struct CdfnorminvD {
    static constexpr const int n_i = 1;
    static constexpr const int n_c = 0;
    static constexpr const int n_o = 1;

    using TypeIn = double;
    using TypeC = void;
    using TypeOut = double;

    CUDA_DEVICE
    Status operator()(ScalarArgs& sa) const {
        double s_a = sa.template get_i<double, 0>();
        double s_y = double(0);

        double t1 = s_a;
        double t2 = fma(2.0, t1, -1.0);
        double t3 = erfinv(t2);
        double t4 = 0x1.6A09E667F3BCC908p+1 * t3;
        s_y = t4;

        sa.set_o<double, 0>(s_y);
        return Status::kSuccess;
    }

    CUDA_DEVICE
    Status math_check(Status e_result, ScalarArgs const& sa) const { 
        double s_a = sa.template get_i<double, 0>();

        if (is_over_absolute_one(s_a)) { return Status::kDomainError; }
        if (is_absolute_one(s_a)) { return Status::kSingularity; }
        return e_result; 
    }
};

struct Erf {
    static constexpr const int n_i = 1;
    static constexpr const int n_c = 0;
    static constexpr const int n_o = 1;

    CUDA_DEVICE
    Status math_check(Status e_result, ScalarArgs const& sa) const { return e_result; }
};

struct Pow {
    static constexpr const int n_i = 2;
    static constexpr const int n_c = 0;
    static constexpr const int n_o = 1;

    CUDA_DEVICE
    Status math_check(Status e_result, ScalarArgs const& sa) const { return e_result; }
};

struct PowH: public Pow {
    using TypeIn = __half;
    using TypeC = void;
    using TypeOut = __half;

    CUDA_DEVICE
    Status operator()(ScalarArgs& sa) const {
        __half s_a = sa.template get_i<__half, 0>();
        __half s_b = sa.template get_i<__half, 1>();
        __half s_y = __half(0);

//--------------------------------------
        float t1 = s_a;
        float t2 = s_b;
        float t3 = powf(t1, t2);
        float iar = t3;
//--------------------------------------

        s_y = static_cast<__half>(iar);

        if (is_inf(s_y) && !is_inf(iar)) { return Status::kOverflow; }
        if (is_zero(s_y) && !is_zero(iar)) { return Status::kUnderflow; }

        sa.set_o<__half, 0>(s_y);
        return Status::kSuccess;
    }
};


struct PowS : public Pow {
    using TypeIn = float;
    using TypeC = void;
    using TypeOut = float;

    CUDA_DEVICE
    Status operator()(ScalarArgs& sa) const {
        float s_a = sa.template get_i<float, 0>();
        float s_b = sa.template get_i<float, 1>();
        float s_y = float(0);

        s_y = powf(s_a, s_b);

        sa.set_o<float, 0>(s_y);
        return Status::kSuccess;
    }
};

struct PowD : public Pow {
    using TypeIn = double;
    using TypeC = void;
    using TypeOut = double;

    CUDA_DEVICE
    Status operator()(ScalarArgs& sa) const {
        double s_a = sa.template get_i<double, 0>();
        double s_b = sa.template get_i<double, 1>();
        double s_y = double(0);

        s_y = pow(s_a, s_b);

        sa.set_o<double, 0>(s_y);
        return Status::kSuccess;
    }
};

struct PowC : public Pow {
    using TypeIn = std::complex<float>;
    using TypeC = void;
    using TypeOut = std::complex<float>;


    CUDA_DEVICE
    Status operator()(ScalarArgs& sa) const {
        std::complex<float> s_a = sa.template get_i<std::complex<float>, 0>();
        std::complex<float> s_b = sa.template get_i<std::complex<float>, 1>();
        std::complex<float> s_y = std::complex<float>(0);

        double t1 = s_a.real();
        double t2 = s_a.imag();
        double t3 = s_b.real();
        double t4 = s_b.imag();

        double t5 = atan2(t2, t1); // phi
        double t6 = hypot(t1, t2); // R
        double t7 = log(t6);

        double t8 = t3 * t7;
        double t9 = t4 * t5;
        double t10 = t3 * t5;
        double t11 = t4 * t7;
        double t12 = t8 - t9;
        double t13 = t10 + t11;

        double t14 = exp(t12);
        double t15 = t14 * cos(t13);
        double t16 = t14 * sin(t13);

        s_y = std::complex<float>(t15, t16);

        sa.set_o<std::complex<float>, 0>(s_y);
        return Status::kSuccess;
    }
};

struct PowZ : public Pow {
    using TypeIn = std::complex<double>;
    using TypeC = void;
    using TypeOut = std::complex<double>;

    CUDA_DEVICE
    Status operator()(ScalarArgs& sa) const {
        std::complex<double> s_a = sa.template get_i<std::complex<double>, 0>();
        std::complex<double> s_b = sa.template get_i<std::complex<double>, 1>();
        std::complex<double> s_y = std::complex<double>(0);

        double t1 = s_a.real();
        double t2 = s_a.imag();
        double t3 = s_b.real();
        double t4 = s_b.imag();

        double t5 = atan2(t2, t1);
        double t6 = hypot(t1, t2);
        double t7 = log(t6);

        double t8 = t3 * t7;
        double t9 = t4 * t5;
        double t10 = t3 * t5;
        double t11 = t4 * t7;
        double t12 = t8 - t9;
        double t13 = t10 + t11;

        double t14 = exp(t12);
        double t15 = t14 * cos(t13);
        double t16 = t14 * sin(t13);

        s_y = std::complex<double>(t15, t16);
 
        sa.set_o<std::complex<double>, 0>(s_y);
        return Status::kSuccess;
    }
};

} // namespace vecma::detail::scalar

