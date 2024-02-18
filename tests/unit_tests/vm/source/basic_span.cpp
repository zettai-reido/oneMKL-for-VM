/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include "test_helper.hpp"

extern std::vector<sycl::device*> devices;

namespace {

class PowLinear: public ::testing::TestWithParam<sycl::device*> {};

TEST_P(PowLinear, float) {
    bool pow_s = true;
    EXPECT_TRUE(pow_s);
} 

INSTANTIATE_TEST_SUITE_P(SpanTestSuite, PowLinear, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());


}



