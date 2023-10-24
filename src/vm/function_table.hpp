///===--- function_table.hpp --- Vector Math function table --===///
/**************************************************************************
* Copyright 2024 Intel Corporation
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
* 
**************************************************************************/

#pragma once

#include <complex>
#include <cstdint>

#include <sycl/sycl.hpp>


#include "oneapi/mkl/types.hpp"

#include "oneapi/mkl/vm/decls.hpp"


namespace oneapi::mkl::vm {
typedef struct {
    int version;

    // API: buffer
    // function: erf
    sycl::event (*erf_h_buffer_linear)(int64_t n, sycl::buffer<sycl::half, 1>& buf_a, sycl::buffer<sycl::half, 1>& buf_y, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_h_buffer_strided)(sycl::buffer<sycl::half, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<sycl::half, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_s_buffer_linear)(int64_t n, sycl::buffer<float, 1>& buf_a, sycl::buffer<float, 1>& buf_y, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_s_buffer_strided)(sycl::buffer<float, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<float, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_d_buffer_linear)(int64_t n, sycl::buffer<double, 1>& buf_a, sycl::buffer<double, 1>& buf_y, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_d_buffer_strided)(sycl::buffer<double, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<double, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode);


    // function: cdfnorminv
    sycl::event (*cdfnorminv_h_buffer_linear)(int64_t n, sycl::buffer<sycl::half, 1>& buf_a, sycl::buffer<sycl::half, 1>& buf_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*cdfnorminv_h_buffer_strided)(sycl::buffer<sycl::half, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<sycl::half, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*cdfnorminv_s_buffer_linear)(int64_t n, sycl::buffer<float, 1>& buf_a, sycl::buffer<float, 1>& buf_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*cdfnorminv_s_buffer_strided)(sycl::buffer<float, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<float, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*cdfnorminv_d_buffer_linear)(int64_t n, sycl::buffer<double, 1>& buf_a, sycl::buffer<double, 1>& buf_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);
    sycl::event (*cdfnorminv_d_buffer_strided)(sycl::buffer<double, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<double, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);


    // function: pow
    sycl::event (*pow_h_buffer_linear)(int64_t n, sycl::buffer<sycl::half, 1>& buf_a, sycl::buffer<sycl::half, 1>& buf_b, sycl::buffer<sycl::half, 1>& buf_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*pow_h_buffer_strided)(sycl::buffer<sycl::half, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<sycl::half, 1>& buf_b, oneapi::mkl::slice sl_b, sycl::buffer<sycl::half, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*pow_s_buffer_linear)(int64_t n, sycl::buffer<float, 1>& buf_a, sycl::buffer<float, 1>& buf_b, sycl::buffer<float, 1>& buf_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*pow_s_buffer_strided)(sycl::buffer<float, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<float, 1>& buf_b, oneapi::mkl::slice sl_b, sycl::buffer<float, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*pow_d_buffer_linear)(int64_t n, sycl::buffer<double, 1>& buf_a, sycl::buffer<double, 1>& buf_b, sycl::buffer<double, 1>& buf_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);
    sycl::event (*pow_d_buffer_strided)(sycl::buffer<double, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<double, 1>& buf_b, oneapi::mkl::slice sl_b, sycl::buffer<double, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);
    sycl::event (*pow_c_buffer_linear)(int64_t n, sycl::buffer<std::complex<float>, 1>& buf_a, sycl::buffer<std::complex<float>, 1>& buf_b, sycl::buffer<std::complex<float>, 1>& buf_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<float>> const& eh);
    sycl::event (*pow_c_buffer_strided)(sycl::buffer<std::complex<float>, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<std::complex<float>, 1>& buf_b, oneapi::mkl::slice sl_b, sycl::buffer<std::complex<float>, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<float>> const& eh);
    sycl::event (*pow_z_buffer_linear)(int64_t n, sycl::buffer<std::complex<double>, 1>& buf_a, sycl::buffer<std::complex<double>, 1>& buf_b, sycl::buffer<std::complex<double>, 1>& buf_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<double>> const& eh);
    sycl::event (*pow_z_buffer_strided)(sycl::buffer<std::complex<double>, 1>& buf_a, oneapi::mkl::slice sl_a, sycl::buffer<std::complex<double>, 1>& buf_b, oneapi::mkl::slice sl_b, sycl::buffer<std::complex<double>, 1>& buf_y, oneapi::mkl::slice sl_y, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<double>> const& eh);


    // API: usm
    // function: erf
    sycl::event (*erf_h_usm_linear)(sycl::queue& queue, int64_t n, const sycl::half* a, sycl::half* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_h_usm_strided)(sycl::queue& queue, const sycl::half* a, oneapi::mkl::slice sl_a, sycl::half* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_s_usm_linear)(sycl::queue& queue, int64_t n, const float* a, float* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_s_usm_strided)(sycl::queue& queue, const float* a, oneapi::mkl::slice sl_a, float* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_d_usm_linear)(sycl::queue& queue, int64_t n, const double* a, double* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_d_usm_strided)(sycl::queue& queue, const double* a, oneapi::mkl::slice sl_a, double* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode);


    // function: cdfnorminv
    sycl::event (*cdfnorminv_h_usm_linear)(sycl::queue& queue, int64_t n, const sycl::half* a, sycl::half* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*cdfnorminv_h_usm_strided)(sycl::queue& queue, const sycl::half* a, oneapi::mkl::slice sl_a, sycl::half* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*cdfnorminv_s_usm_linear)(sycl::queue& queue, int64_t n, const float* a, float* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*cdfnorminv_s_usm_strided)(sycl::queue& queue, const float* a, oneapi::mkl::slice sl_a, float* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*cdfnorminv_d_usm_linear)(sycl::queue& queue, int64_t n, const double* a, double* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);
    sycl::event (*cdfnorminv_d_usm_strided)(sycl::queue& queue, const double* a, oneapi::mkl::slice sl_a, double* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);


    // function: pow
    sycl::event (*pow_h_usm_linear)(sycl::queue& queue, int64_t n, const sycl::half* a, const sycl::half* b, sycl::half* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*pow_h_usm_strided)(sycl::queue& queue, const sycl::half* a, oneapi::mkl::slice sl_a, const sycl::half* b, oneapi::mkl::slice sl_b, sycl::half* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*pow_s_usm_linear)(sycl::queue& queue, int64_t n, const float* a, const float* b, float* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*pow_s_usm_strided)(sycl::queue& queue, const float* a, oneapi::mkl::slice sl_a, const float* b, oneapi::mkl::slice sl_b, float* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*pow_d_usm_linear)(sycl::queue& queue, int64_t n, const double* a, const double* b, double* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);
    sycl::event (*pow_d_usm_strided)(sycl::queue& queue, const double* a, oneapi::mkl::slice sl_a, const double* b, oneapi::mkl::slice sl_b, double* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);
    sycl::event (*pow_c_usm_linear)(sycl::queue& queue, int64_t n, const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<float>> const& eh);
    sycl::event (*pow_c_usm_strided)(sycl::queue& queue, const std::complex<float>* a, oneapi::mkl::slice sl_a, const std::complex<float>* b, oneapi::mkl::slice sl_b, std::complex<float>* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<float>> const& eh);
    sycl::event (*pow_z_usm_linear)(sycl::queue& queue, int64_t n, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<double>> const& eh);
    sycl::event (*pow_z_usm_strided)(sycl::queue& queue, const std::complex<double>* a, oneapi::mkl::slice sl_a, const std::complex<double>* b, oneapi::mkl::slice sl_b, std::complex<double>* y, oneapi::mkl::slice sl_y, sycl_event_vector const& depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<double>> const& eh);


    // API: span
    // function: erf
    sycl::event (*erf_h_span_linear)(sycl::queue& queue, sycl::span<const sycl::half> sp_a, sycl::span<sycl::half> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_h_span_strided)(sycl::queue& queue, sycl::span<const sycl::half> sp_a, oneapi::mkl::slice sl_a, sycl::span<sycl::half> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_s_span_linear)(sycl::queue& queue, sycl::span<const float> sp_a, sycl::span<float> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_s_span_strided)(sycl::queue& queue, sycl::span<const float> sp_a, oneapi::mkl::slice sl_a, sycl::span<float> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_d_span_linear)(sycl::queue& queue, sycl::span<const double> sp_a, sycl::span<double> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode);
    sycl::event (*erf_d_span_strided)(sycl::queue& queue, sycl::span<const double> sp_a, oneapi::mkl::slice sl_a, sycl::span<double> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode);


    // function: cdfnorminv
    sycl::event (*cdfnorminv_h_span_linear)(sycl::queue& queue, sycl::span<const sycl::half> sp_a, sycl::span<sycl::half> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*cdfnorminv_h_span_strided)(sycl::queue& queue, sycl::span<const sycl::half> sp_a, oneapi::mkl::slice sl_a, sycl::span<sycl::half> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*cdfnorminv_s_span_linear)(sycl::queue& queue, sycl::span<const float> sp_a, sycl::span<float> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*cdfnorminv_s_span_strided)(sycl::queue& queue, sycl::span<const float> sp_a, oneapi::mkl::slice sl_a, sycl::span<float> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*cdfnorminv_d_span_linear)(sycl::queue& queue, sycl::span<const double> sp_a, sycl::span<double> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);
    sycl::event (*cdfnorminv_d_span_strided)(sycl::queue& queue, sycl::span<const double> sp_a, oneapi::mkl::slice sl_a, sycl::span<double> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);


    // function: pow
    sycl::event (*pow_h_span_linear)(sycl::queue& queue, sycl::span<const sycl::half> sp_a, sycl::span<const sycl::half> sp_b, sycl::span<sycl::half> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*pow_h_span_strided)(sycl::queue& queue, sycl::span<const sycl::half> sp_a, oneapi::mkl::slice sl_a, sycl::span<const sycl::half> sp_b, oneapi::mkl::slice sl_b, sycl::span<sycl::half> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<sycl::half> const& eh);
    sycl::event (*pow_s_span_linear)(sycl::queue& queue, sycl::span<const float> sp_a, sycl::span<const float> sp_b, sycl::span<float> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*pow_s_span_strided)(sycl::queue& queue, sycl::span<const float> sp_a, oneapi::mkl::slice sl_a, sycl::span<const float> sp_b, oneapi::mkl::slice sl_b, sycl::span<float> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<float> const& eh);
    sycl::event (*pow_d_span_linear)(sycl::queue& queue, sycl::span<const double> sp_a, sycl::span<const double> sp_b, sycl::span<double> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);
    sycl::event (*pow_d_span_strided)(sycl::queue& queue, sycl::span<const double> sp_a, oneapi::mkl::slice sl_a, sycl::span<const double> sp_b, oneapi::mkl::slice sl_b, sycl::span<double> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<double> const& eh);
    sycl::event (*pow_c_span_linear)(sycl::queue& queue, sycl::span<const std::complex<float>> sp_a, sycl::span<const std::complex<float>> sp_b, sycl::span<std::complex<float>> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<float>> const& eh);
    sycl::event (*pow_c_span_strided)(sycl::queue& queue, sycl::span<const std::complex<float>> sp_a, oneapi::mkl::slice sl_a, sycl::span<const std::complex<float>> sp_b, oneapi::mkl::slice sl_b, sycl::span<std::complex<float>> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<float>> const& eh);
    sycl::event (*pow_z_span_linear)(sycl::queue& queue, sycl::span<const std::complex<double>> sp_a, sycl::span<const std::complex<double>> sp_b, sycl::span<std::complex<double>> sp_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<double>> const& eh);
    sycl::event (*pow_z_span_strided)(sycl::queue& queue, sycl::span<const std::complex<double>> sp_a, oneapi::mkl::slice sl_a, sycl::span<const std::complex<double>> sp_b, oneapi::mkl::slice sl_b, sycl::span<std::complex<double>> sp_y, oneapi::mkl::slice sl_y, sycl_event_vector depends, oneapi::mkl::vm::mode mode, oneapi::mkl::vm::error_handler<std::complex<double>> const& eh);


} vm_function_table_t;
} // namespace oneapi::mkl::vm
