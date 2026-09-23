cmake_minimum_required(VERSION 3.22...3.31)
project(jax_fused_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;90")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(CUDAToolkit REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
  OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Found jaxlib include dir: ${JAXLIB_INCLUDE_DIR}")

include(FetchContent)
message(STATUS "Fetching NVIDIA CUTLASS (CuTe)...")
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG        v4.8.0
  GIT_SHALLOW    TRUE
)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable CUTLASS Headers Only" FORCE)
FetchContent_MakeAvailable(cutlass)

add_library(jax_fused_ops SHARED
  src/ffi_bindings.cc
  # src/flash_attention.cu
)

target_include_directories(jax_fused_ops PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${cutlass_SOURCE_DIR}/include
  ${cutlass_SOURCE_DIR}/tools/util/include
  ${Python3_INCLUDE_DIRS}
  ${JAXLIB_INCLUDE_DIR}
)

target_link_libraries(jax_fused_ops PRIVATE
  CUDA::cudart
)

target_compile_options(jax_fused_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -Xptxas=-v
    -Xcompiler=-Wall
    -Xcompiler=-fPIC
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -Wall
    -O3
  >
)

set_target_properties(jax_fused_ops PROPERTIES
  PREFIX ""
  OUTPUT_NAME "libjax_fused_ops"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/jax_fused_ops"
)

message(STATUS "Configured jax_fused_ops shared library.")