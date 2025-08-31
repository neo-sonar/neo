project(neosonar.neo-tests)

FetchContent_Declare(Catch2 GIT_REPOSITORY "https://github.com/catchorg/Catch2" GIT_TAG "v3.10.0" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
set_target_properties(Catch2 PROPERTIES UNITY_BUILD ON)

add_executable(neo-tests)
target_link_libraries(neo-tests PRIVATE neosonar::neo neosonar::compiler_warnings Catch2::Catch2WithMain)
catch_discover_tests(neo-tests TEST_SPEC "--order lex" WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

target_sources(neo-tests
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src/neo/algorithm/allclose_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/algorithm/allmatch_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/algorithm/copy_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/algorithm/mean_squared_error_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/algorithm/mean_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/algorithm/multiply_add_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/algorithm/normalize_energy_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/algorithm/standard_deviation_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/algorithm/variance_test.cpp"

        "${CMAKE_SOURCE_DIR}/src/neo/bit/bit_log2_test.cpp"

        "${CMAKE_SOURCE_DIR}/src/neo/complex/complex_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/complex/scalar_complex_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/complex/split_complex_test.cpp"

        "${CMAKE_SOURCE_DIR}/src/neo/container/compressed_accessor_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/container/csr_matrix_test.cpp"

        "${CMAKE_SOURCE_DIR}/src/neo/convolution/compressed_fdl_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/convolution/dense_fdl_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/convolution/direct_convolve_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/convolution/fdl_index_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/convolution/fft_convolver_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/convolution/normalize_impulse_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/convolution/overlap_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/convolution/uniform_partition_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/convolution/uniform_partitioned_convolver_test.cpp"

        "${CMAKE_SOURCE_DIR}/src/neo/fft/dct_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/fft/dft_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/fft/rfftfreq_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/fft/fft_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/fft/rfft_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/fft/split_fft_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/fft/stft_test.cpp"

        "${CMAKE_SOURCE_DIR}/src/neo/fixed_point/fixed_point_test.cpp"

        "${CMAKE_SOURCE_DIR}/src/neo/math/a_weighting_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/math/abs_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/math/imag_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/math/ipow_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/math/log2_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/math/real_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/math/windowing_test.cpp"

        "${CMAKE_SOURCE_DIR}/src/neo/simd_test.cpp"

        "${CMAKE_SOURCE_DIR}/src/neo/unit/decibel_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/neo/unit/mel_test.cpp"
)
