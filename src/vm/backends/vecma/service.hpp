#pragma once

#include <string>

#include "detailed_mode.hh"
#include "enums.hh"
#include "xtypes.hh"

namespace yma::detail {

DetailedMode get_actual_detailed_mode(yma::mode explicit_mode);
std::string stringify(Status status);
std::string stringify(DetailedMode const& dm);
int32_t get_block_size(Opcode op);

} // namespace yma::detail

