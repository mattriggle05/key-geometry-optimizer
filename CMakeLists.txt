cmake_minimum_required(VERSION 3.18)
project(key_geometry_optimizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(key_geometry_optimizer
    src/main.cpp
    src/kernels.cu
    src/fields.cpp
)

target_link_libraries(key_geometry_optimizer PRIVATE curand)

set_target_properties(key_geometry_optimizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(key_geometry_optimizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
)