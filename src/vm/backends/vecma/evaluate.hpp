#pragma once

#include "error_handler.hpp"
#include "strider.hpp"
#include "vector_args.hpp"

namespace vecma::detail {

Status evaluate_pow_h(VectorArgs& va, Strider str, ErrorHandler const& eh);
Status evaluate_pow_s(VectorArgs& va, Strider str, ErrorHandler const& eh);
Status evaluate_pow_d(VectorArgs& va, Strider str, ErrorHandler const& eh);
Status evaluate_pow_c(VectorArgs& va, Strider str, ErrorHandler const& eh);
Status evaluate_pow_z(VectorArgs& va, Strider str, ErrorHandler const& eh);


} // namespace vecma::detail

