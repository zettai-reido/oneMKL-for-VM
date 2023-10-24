#==========================================================================
#  Copyright (C) Codeplay Software Limited
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  For your convenience, a copy of the License has been included in this
#  repository.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#=========================================================================

set(LIB_NAME onemkl_vm_vecma)

set(LIB_OBJ ${LIB_NAME}_obj)
set(CU_LIB_OBJ ${LIB_NAME}_cu_obj)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

find_package(CUDAToolkit)
enable_language(CUDA)

set(SOURCES service.cpp iface.cpp
     $<$<BOOL:${BUILD_SHARED_LIBS}>:vecma_wrappers.cpp>)

set(CU_SOURCES evaluate.cu
)

add_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT ${SOURCES})
add_library(${CU_LIB_OBJ} OBJECT ${CU_SOURCES})

target_include_directories(${LIB_OBJ}
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src/include
          ${PROJECT_SOURCE_DIR}/src
          ${CMAKE_BINARY_DIR}/bin
)

target_include_directories(${CU_LIB_OBJ}
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src/include
          ${PROJECT_SOURCE_DIR}/src
          ${CMAKE_BINARY_DIR}/bin
)

target_compile_features(${LIB_OBJ} PUBLIC cxx_std_17)
target_compile_features(${CU_LIB_OBJ} PUBLIC cxx_std_17)

target_compile_options(${LIB_OBJ}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ferror-limit=1>
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)
target_compile_options(${CU_LIB_OBJ}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="-ferror-limit=1">
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wall>
)


set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

set_target_properties(${CU_LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON)


if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${LIB_OBJ} SOURCES ${SOURCES})
endif()

target_link_libraries(${LIB_OBJ}
  PUBLIC ONEMKL::SYCL::SYCL
)

target_link_libraries(${CU_LIB_OBJ}
  PRIVATE CUDA::cudart
)

target_link_libraries(${LIB_NAME} PUBLIC ${LIB_OBJ} ${CU_LIB_OBJ})

# Add major version to the library
set_target_properties(${LIB_NAME} PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Add dependencies rpath to the library
list(APPEND CMAKE_BUILD_RPATH $<TARGET_FILE_DIR:${LIB_NAME}>)

# Add the library to install package
install(TARGETS ${LIB_OBJ} EXPORT oneMKLTargets)
install(TARGETS ${CU_LIB_OBJ} EXPORT oneMKLTargets)

install(TARGETS ${LIB_NAME} EXPORT oneMKLTargets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

