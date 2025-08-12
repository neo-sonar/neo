cmake_minimum_required(VERSION 3.26...3.30)
project(neosonar-neo-dev VERSION 0.1.0)

option(NEO_ENABLE_BENCHMARKS "Build benchmarks" ON)
option(NEO_ENABLE_PLUGIN "Build audio plugin" ON)
option(NEO_ENABLE_TESTS "Build tests" ON)
option(NEO_ENABLE_TOOLS "Build tools" ON)

option(NEO_ENABLE_APPLE_ACCELERATE "Link against Apple Accelerate" ON)
option(NEO_ENABLE_INTEL_IPP "Link against Intel IPP" OFF)
option(NEO_ENABLE_INTEL_MKL "Link against Intel MKL" OFF)
option(NEO_ENABLE_XSIMD "Link against xsimd" ON)

find_program(CCACHE ccache)
if (CCACHE)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif ()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()
include(FetchContent)
include(cmake/compiler_warnings.cmake)

FetchContent_Declare(mdspan GIT_REPOSITORY "https://github.com/kokkos/mdspan" GIT_TAG "stable" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(mdspan)

if(NEO_ENABLE_INTEL_MKL)
    set(MKL_LINK "static")
    set(MKL_THREADING "sequential")
endif()

if(NEO_ENABLE_XSIMD)
    FetchContent_Declare(xsimd GIT_REPOSITORY "https://github.com/xtensor-stack/xsimd" GIT_TAG "13.2.0" GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(xsimd)
endif()

add_subdirectory(src)

if(DEFINED SKBUILD)
    add_subdirectory(extra/python)
    return()
endif()

if(NEO_ENABLE_TESTS)
    add_subdirectory(test)
endif()

if(NEO_ENABLE_BENCHMARKS)
    add_subdirectory(extra/benchmark)
endif()

if(NEO_ENABLE_TOOLS)
    add_subdirectory(extra/cli)
endif()

if(NEO_ENABLE_PLUGIN)
    add_subdirectory(extra/plugin)
endif()
