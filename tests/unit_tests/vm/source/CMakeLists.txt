#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
#===============================================================================

#Build object from all test sources
set(VM_SOURCES
    basic_span.cpp
)

if(BUILD_SHARED_LIBS)
  add_library(vm_source_rt OBJECT ${VM_SOURCES})
  target_compile_options(vm_source_rt PRIVATE -DCALL_RT_API)
  target_include_directories(vm_source_rt
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
  )
  target_link_libraries(vm_source_rt PUBLIC ONEMKL::SYCL::SYCL)
endif()

add_library(vm_source_ct OBJECT ${VM_SOURCES})
target_include_directories(vm_source_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
)
target_link_libraries(vm_source_ct PUBLIC ONEMKL::SYCL::SYCL)
