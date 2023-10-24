#ifndef ONEAPI_MKL_VM
#define ONEAPI_MKL_VM 1

#include <sycl/sycl.hpp>

#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl/vm/decls.hpp"

#ifdef ENABLE_MKLCPU_BACKEND
#include "oneapi/mkl/vm/detail/mklcpu/vm_ct.hpp"
#endif

#ifdef ENABLE_MKLGPU_BACKEND
#include "oneapi/mkl/vm/detail/mklgpu/vm_ct.hpp"
#endif

#ifdef ENABLE_VECMA_BACKEND
#include "oneapi/mkl/vm/detail/vecma/vm_ct.hpp"
#endif

#include "oneapi/mkl/vm/detail/vm_rt.hpp"

#endif // #ifndef ONEAPI_MKL_VM 1


