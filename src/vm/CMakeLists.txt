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

# Build backends
add_subdirectory(backends)

# Recipe for VM loader object
if(BUILD_SHARED_LIBS)
add_library(onemkl_vm OBJECT)
target_sources(onemkl_vm PRIVATE vm_loader.cpp)
target_include_directories(onemkl_vm
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src
          ${PROJECT_SOURCE_DIR}/src/include
          ${CMAKE_BINARY_DIR}/bin
          ${ONEMKL_GENERATED_INCLUDE_PATH}
          $<TARGET_FILE_DIR:onemkl>
)

target_compile_options(onemkl_vm PRIVATE ${ONEMKL_BUILD_COPT})

set_target_properties(onemkl_vm PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(onemkl_vm PUBLIC ONEMKL::SYCL::SYCL)
endif()

