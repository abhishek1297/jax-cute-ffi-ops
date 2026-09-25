cmake_minimum_required(VERSION 3.22)
project(jax_ffi_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;90")
endif()

# -----------------------------------------------------------------------------
# Global Dependencies
# -----------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(CUDAToolkit REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
  OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Fetch CUTLASS
include(FetchContent)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG        v4.8.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cutlass)

# -----------------------------------------------------------------------------
# Add Subdirectories
# -----------------------------------------------------------------------------
add_subdirectory(src/add_one)
add_subdirectory(src/flash_attention)