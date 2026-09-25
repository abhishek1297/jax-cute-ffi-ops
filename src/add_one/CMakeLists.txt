file(GLOB CANDIDATE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

add_library(add_one SHARED
  ${CANDIDATE_SOURCES}
)

target_include_directories(add_one SYSTEM PRIVATE
  ${JAXLIB_INCLUDE_DIR}
)

target_include_directories(add_one PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Python3_INCLUDE_DIRS}
  ${cutlass_SOURCE_DIR}/include
  ${cutlass_SOURCE_DIR}/tools/util/include
)

target_link_libraries(add_one PRIVATE
  CUDA::cudart
)

if(TARGET cutlass::cutlass)
  target_link_libraries(add_one PRIVATE cutlass::cutlass)
elseif(TARGET nvidia_cutlass)
  target_link_libraries(add_one PRIVATE nvidia_cutlass)
endif()

target_compile_options(add_one PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -Xptxas=-v
    -Xcompiler=-Wall
    -Xcompiler=-fPIC
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -Wall
    -O3
    -Wno-sign-compare
    -Wno-attributes
  >
)

set_target_properties(add_one PROPERTIES
  PREFIX ""
  OUTPUT_NAME "libadd_one"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
)