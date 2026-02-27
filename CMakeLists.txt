cmake_minimum_required(VERSION 3.16)

project(debug LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(TinyXML2 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/fk_triangulation)
add_executable(debug main3D.cu src/loader.cu)

set_target_properties(debug PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(debug PUBLIC tinyxml2::tinyxml2)
set_target_properties(debug PROPERTIES
    CUDA_ARCHITECTURES 75
)
target_compile_options(debug PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
