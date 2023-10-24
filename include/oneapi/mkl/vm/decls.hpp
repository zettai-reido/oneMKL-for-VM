/*******************************************************************************
* Copyright 2019-2023 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#ifndef ONEAPI_MKL_VM_DECLS_HPP
#define ONEAPI_MKL_VM_DECLS_HPP 1

#include <complex>
#include <cstdint>
#include <exception>

#include <sycl/sycl.hpp>

#ifdef MKL_BUILD_DLL
#define ONEAPI_MKL_VM_EXPORT __declspec(dllexport)
#define ONEAPI_MKL_VM_EXPORT_CPP __declspec(dllexport)
#else
#define ONEAPI_MKL_VM_EXPORT extern
#define ONEAPI_MKL_VM_EXPORT_CPP
#endif

namespace oneapi::mkl::vm {

namespace detail {

using EventVector = std::vector<sycl::event>;

enum class Mode : std::uint32_t {
  kNotDefined = 0x0,

  kHa = 0x4,
  kLa = 0xC,
  kEp = 0xF,
  kAccuracyMask = 0xF,

  kGlobalStatusReport = 0x100,
  kGlobalStatusQuiet = 0x200,
  kGlobalStatusMask = 0x300,

  kSliceNormal = 0x1000,
  kSliceMinimum = 0x2000,
  kSliceCyclic = 0x4000,

  kBadargException = 0x1'0000,
  kBadargQuiet = 0x2'0000,
  kBadargMask = 0x3'0000,

  kFallbackEnabled = 0x10'0000,
  kFallbackWarning = 0x20'0000,
  kFallbackException = 0x40'0000,
  kFallbackPermissive = 0x80'0000,
  kFallbackMask = 0xF0'0000,

  kVerboseQuiet = 0x100'0000,
  kVerboseSubmit = 0x200'0000,
  kVerboseCall = 0x400'0000,
  kVerboseMask = 0x700'0000,

  kDefault = (kHa | kGlobalStatusQuiet | kSliceNormal | kBadargException | kFallbackEnabled | kVerboseQuiet),

// user interface is lowercase
  not_defined = kNotDefined,

  ha = kHa,
  la = kLa,
  ep = kEp,

  global_status_report = kGlobalStatusReport,
  global_status_quiet = kGlobalStatusQuiet,

  slice_normal = kSliceNormal,
  slice_minimum = kSliceMinimum,
  slice_cyclic = kSliceCyclic,

  badarg_exception = kBadargException,
  badarg_quiet = kBadargQuiet,

  fallback_enabled = kFallbackEnabled,
  fallback_warning = kFallbackWarning,
  fallback_exception = kFallbackException,
  fallback_permissive = kFallbackPermissive,
  fallback_mask = kFallbackMask,

  verbose_quiet = kVerboseQuiet,
  verbose_submit = kVerboseSubmit,
  verbose_call = kVerboseCall
}; // enum class Mode

enum class Status : std::uint32_t {
  kNotDefined = 0x0,
  kSuccess = 0x0,
  kDomainError = 0x1,
  kSingularity = 0x2,
  kOverflow = 0x4,
  kUnderflow = 0x8,
  kAccuracyWarning = 0x80,
  
  kFixAll = ( kDomainError | kSingularity | kOverflow | kUnderflow | kAccuracyWarning ),

  kComputationEmpty = 0x100,
  kRuntimeError = 0xE000'0000,

  not_defined = kNotDefined,
  success = kSuccess,
  errdom = kDomainError,
  sing = kSingularity,
  overflow = kOverflow,
  underflow = kUnderflow,
  accuracy_warning = kAccuracyWarning,
  fix_all = kFixAll,

  computation_empty = kComputationEmpty,
  runtime_error = kRuntimeError // special status for non-MKL backend (i.e. VECMA)
};

template <typename BitType> struct bits_enabled {  static constexpr bool value = false; };
template <> struct bits_enabled<Mode> { static constexpr bool value = true; };
template <> struct bits_enabled<Status> { static constexpr bool value = true; };

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, BitType>
operator|(BitType lhs, BitType rhs) {
    using U = std::underlying_type_t<BitType>;
  auto r = static_cast<typename std::underlying_type_t<BitType>>(lhs) |
           static_cast<typename std::underlying_type_t<BitType>>(rhs);
  return static_cast<BitType>(r);
}

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, BitType>&
operator|=(BitType& lhs, BitType rhs) {
  auto r = static_cast<typename std::underlying_type_t<BitType>>(lhs) |
           static_cast<typename std::underlying_type_t<BitType>>(rhs);
  lhs = static_cast<BitType>(r);
  return lhs;
}

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, BitType>
operator&(BitType lhs, BitType rhs) {
  auto r = static_cast<typename std::underlying_type_t<BitType>>(lhs) &
           static_cast<typename std::underlying_type_t<BitType>>(rhs);
  return static_cast<BitType>(r);
}

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, BitType>
operator&=(BitType& lhs, BitType rhs) {
  auto r = static_cast<typename std::underlying_type_t<BitType>>(lhs) &
           static_cast<typename std::underlying_type_t<BitType>>(rhs);
  lhs = static_cast<BitType>(r);
  return lhs;
}

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, BitType>
operator^(BitType lhs, BitType rhs) {
  auto r = static_cast<typename std::underlying_type_t<BitType>>(lhs) ^
           static_cast<typename std::underlying_type_t<BitType>>(rhs);
  return static_cast<BitType>(r);
}

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, BitType>
operator^=(BitType& lhs, BitType rhs) {
  auto r = static_cast<typename std::underlying_type_t<BitType>>(lhs) ^
           static_cast<typename std::underlying_type_t<BitType>>(rhs);
  lhs = static_cast<BitType>(r);
  return lhs;
}

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, bool>
operator!(BitType v) {
  return (0 == static_cast<typename std::underlying_type_t<BitType>>(v));
}

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, bool>
has_none(BitType v, BitType mask) {
  auto r = static_cast<typename std::underlying_type_t<BitType>>(v) &
           static_cast<typename std::underlying_type_t<BitType>>(mask);
  return (0 == r);
}

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, bool>
has_any(BitType v, BitType mask) {
  auto r = static_cast<typename std::underlying_type_t<BitType>>(v) &
           static_cast<typename std::underlying_type_t<BitType>>(mask);
  return (0 != r);
}

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, bool>
has_all(BitType v, BitType mask) {
  auto r = static_cast<typename std::underlying_type_t<BitType>>(v) &
           static_cast<typename std::underlying_type_t<BitType>>(mask);
  return (static_cast<typename std::underlying_type_t<BitType>>(mask) == r);
}

template <typename BitType>
constexpr typename std::enable_if_t<bits_enabled<BitType>::enabled, bool>
has_only(BitType v, BitType mask) {
  auto r = static_cast<typename std::underlying_type_t<BitType>>(v) ^
           static_cast<typename std::underlying_type_t<BitType>>(mask);
  return (0 == r);
}

struct ONEAPI_MKL_VM_EXPORT_CPP ErrorHandler_base {
  virtual ~ErrorHandler_base() {}
};

template <typename NumberType>
struct ONEAPI_MKL_VM_EXPORT_CPP ErrorHandler : public ErrorHandler_base {
  bool enabled_;
  bool is_usm_;

  sycl::buffer<Status, 1> buf_status_;
  Status* usm_status_;
  int64_t len_;

  Status status_to_fix_;
  NumberType fixup_value_;

  bool copy_sign_;

  ErrorHandler()
      : enabled_{false}, is_usm_{false},

        buf_status_{sycl::buffer<Status, 1>(1)}, usm_status_{nullptr},
        len_{0}, status_to_fix_{Status::kNotDefined}, fixup_value_{NumberType{}},
        copy_sign_{false} {}

  ErrorHandler(Status status_to_fix, NumberType fixup_value,
                bool copy_sign = false)
      : enabled_{true}, is_usm_{false},

        buf_status_{sycl::buffer<Status, 1>(1)},
        usm_status_{nullptr}, len_{0}, status_to_fix_{status_to_fix},
        fixup_value_{fixup_value}, copy_sign_{copy_sign} {}

  ErrorHandler(Status* array, std::int64_t len = 1,
                Status status_to_fix = Status::kNotDefined,
                NumberType fixup_value = {}, bool copy_sign = false)
      : enabled_{true}, is_usm_{true},

        buf_status_{sycl::buffer<Status, 1>(1)},
        usm_status_{array}, len_{len}, status_to_fix_{status_to_fix},
        fixup_value_{fixup_value}, copy_sign_{copy_sign} {}

  ErrorHandler(sycl::buffer<Status, 1>& buf, std::int64_t len = 1,
                Status status_to_fix = Status::kNotDefined,
                NumberType fixup_value = {}, bool copy_sign = false)
      : enabled_{true}, is_usm_{false},

        buf_status_{buf}, usm_status_{nullptr}, len_{len},
        status_to_fix_{status_to_fix}, fixup_value_{fixup_value},
        copy_sign_{copy_sign} {}
}; // struct ErrorHandler

} // namespace detail

using mode = detail::Mode;
using status = detail::Status;

using sycl_event_vector = detail::EventVector;

template
<typename NumberType>
using error_handler = detail::ErrorHandler<NumberType>;

// Service functions
ONEAPI_MKL_VM_EXPORT oneapi::mkl::vm::mode get_mode(sycl::queue& queue);
ONEAPI_MKL_VM_EXPORT oneapi::mkl::vm::mode
set_mode(sycl::queue& queue, oneapi::mkl::vm::mode new_mode);

ONEAPI_MKL_VM_EXPORT oneapi::mkl::vm::status get_status(sycl::queue& queue);
ONEAPI_MKL_VM_EXPORT oneapi::mkl::vm::status
set_status(sycl::queue& queue, oneapi::mkl::vm::status new_status);
ONEAPI_MKL_VM_EXPORT oneapi::mkl::vm::status clear_status(sycl::queue& queue);

} // namespace oneapi::mkl::vm

#endif // ifndef _ONEAPI_MKL_VM_DECLS_HPP_

