/*******************************************************************************
* Copyright Codeplay Software
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/blas/detail/generic/onemath_blas_generic.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace generic {
namespace column_major {

// BUFFER
void gemm(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<oneapi::math::bfloat16, 1>& a, std::int64_t lda,
          sycl::buffer<oneapi::math::bfloat16, 1>& b, std::int64_t ldb, float beta,
          sycl::buffer<float, 1>& c, std::int64_t ldc) {
    throw unimplemented("blas", "gemm", " for bfloat16");
}

// USM
sycl::event gemm(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                 const oneapi::math::bfloat16* a, std::int64_t lda, const oneapi::math::bfloat16* b,
                 std::int64_t ldb, float beta, float* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

} // namespace column_major
namespace row_major {

// BUFFER
void gemm(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<oneapi::math::bfloat16, 1>& a, std::int64_t lda,
          sycl::buffer<oneapi::math::bfloat16, 1>& b, std::int64_t ldb, float beta,
          sycl::buffer<float, 1>& c, std::int64_t ldc) {
    throw unimplemented("blas", "gemm", " for bfloat16");
}

// USM
sycl::event gemm(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                 const oneapi::math::bfloat16* a, std::int64_t lda, const oneapi::math::bfloat16* b,
                 std::int64_t ldb, float beta, float* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

} // namespace row_major
} // namespace generic
} // namespace blas
} // namespace math
} // namespace oneapi
