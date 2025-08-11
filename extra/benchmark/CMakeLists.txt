project(neosonar-neo-benchmark)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable Google benchmark tests")
FetchContent_Declare(benchmark GIT_REPOSITORY "https://github.com/google/benchmark" GIT_TAG "v1.9.0" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(benchmark)

function(neo_add_benchmark _target)
    add_executable(${_target}_benchmark src/${_target}.cpp)
    target_link_libraries(${_target}_benchmark PRIVATE neosonar::neo benchmark::benchmark)
endfunction(neo_add_benchmark)

neo_add_benchmark(biquad)
neo_add_benchmark(convolution)
neo_add_benchmark(fft)
neo_add_benchmark(memcpy)
neo_add_benchmark(multiply)
neo_add_benchmark(multiply_add)
neo_add_benchmark(rfft)
if(NEO_ENABLE_XSIMD)
    neo_add_benchmark(simd_fft)
endif()

add_executable(fixed_point_benchmark src/fixed_point.cpp)
target_link_libraries(fixed_point_benchmark PRIVATE neosonar::neo)
