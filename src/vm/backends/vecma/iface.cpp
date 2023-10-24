#include "vecma.h"

#include "error_handler.hpp"
#include "evaluate.hpp"
#include "strider.hpp"
#include "xtypes.hpp"
#include "vector_args.hpp"
#include "vector_wrapper.hpp"

namespace vecma::detail {

static_assert(kMaxI == kVECMA_MAX_I, "kMaxI == kVECMA_MAX_I");
static_assert(kMaxO == kVECMA_MAX_O, "kMaxI == kVECMA_MAX_O");
static_assert(kMaxC == kVECMA_MAX_C, "kMaxI == kVECMA_MAX_C");

// enable operations for e_vecma_status
template<> struct BitOperations<e_vecma_status> { static constexpr const bool enabled = true; };

static constexpr Slice convert_slice(s_vecma_slice sek) { return Slice(sek.start, sek.size, sek.stride); }

static constexpr e_vecma_status convert_status(Status s) {
    if (any_of(s, Status::kRuntimeMask)) { return kVECMA_RUNTIME_ERROR; }

    e_vecma_status es = kVECMA_SUCCESS;

    if (any_of(s, Status::kDomainError)) { es |= kVECMA_DOMAIN_ERROR; }
    if (any_of(s, Status::kSingularity)) { es |= kVECMA_SINGULARITY; }
    if (any_of(s, Status::kUnderflow)) { es |= kVECMA_UNDERFLOW; }
    if (any_of(s, Status::kOverflow)) { es |= kVECMA_OVERFLOW; }

    return es;
}

static VectorArgs make_vector_args(struct s_vecma* ekr) {
    return VectorArgs();
}

static Strider make_strider(struct s_vecma* ekr) {
    return Strider();
}

static ErrorHandler make_error_handler(struct s_vecma* ekr) {
    return ErrorHandler();
}

static Mode make_mode(struct e_vecma* ekr) {

}

Status evaluate(e_vecma_function func, struct s_vecma* ekr) {
    Status ret = Status::kSuccess;

    VectorArgs va = make_vector_args(ekr);
    Strider str = make_strider(ekr);
    ErrorHandler eh = make_error_handler(ekr);

    switch (func) {
        case kVECMA_FUNC_POW:
        case kVECMA_FUNC_NOP:
            break;
    }
    return ret;
}

} // namespace vecma::detail

extern "C" {

e_vecma_status
vecma_evaluate(e_vecma_function func, struct s_vecma* ekr) {
    using namespace vecma::detail;

    auto status = evaluate(func, ekr);
    e_vecma_status s = convert_status(status);
    return s;
}

} // extern "C"

