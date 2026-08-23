cmake_minimum_required(VERSION 3.24)

project(cuda_foundations_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

include(CTest)

find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE CUDA_FOUNDATIONS_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

if(NOT CUDA_FOUNDATIONS_GIT_COMMIT)
    set(CUDA_FOUNDATIONS_GIT_COMMIT "unknown")
endif()

function(configure_cuda_target target_name)
    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
    )
endfunction()

add_library(transpose_core OBJECT src/transpose/transpose.cu)
target_compile_definitions(transpose_core PRIVATE CUDA_FOUNDATIONS_TRANSPOSE_CORE_ONLY=1)
configure_cuda_target(transpose_core)

add_executable(transpose src/transpose/transpose.cu)
configure_cuda_target(transpose)

add_executable(transpose_test
    tests/transpose_test.cu
    $<TARGET_OBJECTS:transpose_core>
)
configure_cuda_target(transpose_test)

add_executable(transpose_bench
    benchmarks/transpose_bench.cu
    $<TARGET_OBJECTS:transpose_core>
)
target_compile_definitions(
    transpose_bench
    PRIVATE CUDA_FOUNDATIONS_GIT_COMMIT="${CUDA_FOUNDATIONS_GIT_COMMIT}"
)
configure_cuda_target(transpose_bench)

if(BUILD_TESTING)
    add_test(NAME transpose_correctness COMMAND transpose_test)
endif()
