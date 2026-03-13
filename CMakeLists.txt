cmake_minimum_required(VERSION 3.18)
project(keygeometry_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(keygeometry_test
    src/main.cpp
    src/kernels.cu
    src/fields.cpp
)

target_link_libraries(keygeometry_test PRIVATE curand)

set_target_properties(keygeometry_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(keygeometry_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
)