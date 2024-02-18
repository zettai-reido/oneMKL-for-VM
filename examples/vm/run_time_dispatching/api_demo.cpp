/*******************************************************************************
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
*******************************************************************************/
/*
 *
 *  Content:
 *            demonstration of usage of VM APIs:
 *            for SYCL buffer,
 *            USM shared and device pointers,
 *            ordinary heap and stack pointers,
 *            error handler in replacement(fixup) mode
 *                          on
 *            generation of random normal variable N(0, 1)
 *            using inverse cumulative distribution function
 *
 *******************************************************************************/

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <type_traits>

#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

namespace {

void preamble(sycl::device& dev) {
    std::string dev_name = dev.template get_info<sycl::info::device::name>();
    std::string driver_version = dev.template get_info<sycl::info::device::version>();
    fprintf(stderr, "\t       device name: %s\n", dev_name.c_str());
    fprintf(stderr, "\t    driver version: %s\n\n", driver_version.c_str());
}

void async_sycl_error(sycl::exception_list el) {
    for (auto l = el.begin(); l != el.end(); ++l) {
        try {
            std::rethrow_exception(*l);
        }
        catch (sycl::exception const& e) {
            fprintf(stderr, "async SYCL exception: %s (code: %d)\n", e.what(), e.code().value());
        }
    }
}

struct UniformFiller {
    std::mt19937_64 rng;
    UniformFiller(uint64_t seed) : rng(seed) { }
    double operator()() { return (rng() >> 11) * 0x1p-53; }
};

bool check_mean(double mean, double expected, double sigma, int64_t n) {
    double adiff = std::fabs(mean - expected);
    double err_estimate = sigma / std::sqrt(n * 1.0);
    return (adiff / err_estimate) < 3.0;
}

bool check_stddev(double std_dev, double expected, double sigma, int64_t n) {
    double adiff = std::fabs(std_dev - expected);
    double err_estimate = sigma / std::sqrt(2 * (n - 1.0));
    return (adiff / err_estimate) < 3.0;
}

template <typename T>
bool print_results(const char* method, int64_t n, T* y) {
    double s1 = std::accumulate(y, y + n, 0.0, [=](double s, double t) { return s + t; });
    double s2 = std::accumulate(y, y + n, 0.0, [=](double s, double t) { return s + t * t; });

    double mean = s1 / n;
    double stddev = std::sqrt((s2 - s1 * s1 / n) / (n - 1));

    const char* float_type_string{ (sizeof(T) == 4) ? "float" : "double" };

    auto mean_ok = (check_mean(mean, 0.0, 1.0, n));
    auto stddev_ok = (check_stddev(stddev, 1.0, 1.0, n));

    fprintf(stderr, "%s\n", float_type_string);
    fprintf(stderr, "\t%s\n", method);
    fprintf(stderr, "\t\t%10ld\n", n);
    fprintf(stderr, "\t\t%16.10lf\n", mean, mean_ok ? "( PASS )" : "( FAIL )");
    fprintf(stderr, "\t\t%16.10lf\n", stddev, stddev_ok ? "( PASS )" : "( FAIL )");

    return (mean_ok && stddev_ok);
}

template <typename T>
bool run_usm(int64_t n, sycl::queue& queue) {
    namespace one = oneapi::mkl;

    // user can mix device and shared pointers
    T* a = sycl::malloc_shared<T>(n, queue);
    T* y = new T[n];
    T* dev_y = sycl::malloc_device<T>(n, queue);

    std::generate(a, a + n, UniformFiller(88883)); // shared usm is accessible from host
    std::fill(y, y + n, std::nan(""));

    queue.memcpy(dev_y, y, n * sizeof(T)); // memcpy to device to clean device memory
    queue.wait_and_throw(); // memcpy is async too

    one::vm::cdfnorminv(oneapi::mkl::libkey(queue), queue, n, a, dev_y,
                        { /* no dependent events */ }, one::vm::mode::ha,
                        { one::vm::status::sing, 7.0f });
    queue.wait_and_throw(); // USM call is asynchronous so wait is needed

    queue.memcpy(y, dev_y, n * sizeof(T)); // memcpy back to host
    queue.wait_and_throw(); // memcpy is async too

    auto pass = print_results("on_usm", n, y);

    sycl::free(dev_y, queue);
    delete[] y;
    sycl::free(a, queue);

    return pass;
}

template <typename T>
bool run_buffer(int64_t n, sycl::queue& queue) {
    namespace one = oneapi::mkl;

    T* a = new T[n];
    T* y = new T[n];

    std::generate(a, a + n, UniformFiller(90001));

    {
        sycl::buffer<T, 1> buf_a{
            a, a + n
        }; // SYCL buffer which copies data from 'a', but does not copy back
        sycl::buffer<T, 1> buf_y{ y, n }; // SYCL buffer which copy back to 'y' on destruction

        one::vm::cdfnorminv(oneapi::mkl::backend_selector<Backend>(queue), n, buf_a, buf_y,
                            one::vm::mode::ha, { one::vm::status::sing, 7.0f });

    } // buf_y destructed, data now in 'y'

    auto pass = print_results("on_buffer", n, y);

    delete[] y;
    delete[] a;

    return pass;
}

template <oneapi::mkl::backend Backend, typename T, size_t n>
bool run_stack(sycl::queue& queue) {
    namespace one = oneapi::mkl;

    T a[n];
    T y[n];

    std::generate(a, a + n, UniformFiller(80777));
    std::fill(y, y + n, std::nan(""));

    one::vm::cdfnorminv(oneapi::mkl::backend_selector<Backend>(queue), n, a, y,
                        { /* no dependent events */ }, one::vm::mode::ha,
                        { one::vm::status::sing, 7.0f });
    // function returns with result ready when used with heap pointer as output

    auto pass = print_results("on_stack", n, y);

    return pass;
}

template <oneapi::mkl::backend Backend, typename T>
bool run_heap(int64_t n, sycl::queue& queue) {
    namespace one = oneapi::mkl;

    T* a = new T[n];
    T* y = new T[n];

    std::generate(a, a + n, UniformFiller(99001));
    std::fill(y, y + n, std::nan(""));

    one::vm::cdfnorminv(oneapi::mkl::backend_selector<Backend>(queue), n, a, y,
                        { /* no dependent events */ }, one::vm::mode::ha,
                        { one::vm::status::sing, 7.0f });
    // function returns with result ready when used with heap pointer as output

    auto pass = print_results("on_heap", n, y);

    delete[] y;
    delete[] a;

    return pass;
}

template <oneapi::mkl::backend Backend>
int run_on(sycl::device& dev) {
    bool pass = true;

    double mean = std::nan("");
    double std_dev = std::nan("");

    constexpr int vector_stack_len = 1024;
    int64_t vector_heap_len = 10'000'000;
    int64_t vector_buffer_len = 10'000'000;
    int64_t vector_usm_len = 10'000'000;

    preamble(dev);

    sycl::queue queue(dev, async_sycl_error);

    pass &= run_stack<Backend, float, vector_stack_len>(queue);
    pass &= run_heap<Backend, float>(vector_heap_len, queue);
    pass &= run_buffer<Backend, float>(vector_buffer_len, queue);
    pass &= run_usm<Backend, float>(vector_usm_len, queue);
    pass &= run_stack<Backend, double, vector_stack_len>(queue);
    pass &= run_heap<Backend, double>(vector_heap_len, queue);
    pass &= run_buffer<Backend, double>(vector_buffer_len, queue);
    pass &= run_usm<Backend, double>(vector_usm_len, queue);

    return pass;
} // int run_on(sycl::device &dev)

} // namespace

//
// Main entry point for example.
//
//  For each device selected and each data type supported, the example
//  runs with all supported data types
//
int main(int argc, char** argv) {
    int ret = 0; // return status
    fprintf(stderr, "sycl vm_host_api_demo: started...\n");

    try {
        sycl::device gpu(sycl::gpu_selector_v);
        sycl::device cpu(sycl::cpu_selector_v);
        unsigned int vendor_id = gpu.get_info<sycl::info::device::vendor_id>();
        if (vendor_id != NVIDIA_ID) {
            fprintf(stderr, "GPU is not CUDA-supported\n");
            return -1;
        }
    }
    catch (sycl::exception const& e) {
        fprintf(stderr, "SYCL exception: %s (code: %d)\n", e.what(), e.code().value());
    }

    fprintf(stdout, "vm_host_api_demo: %s\n\n", (ret != 0) ? "FAIL" : "PASS");
    return ret;
}
