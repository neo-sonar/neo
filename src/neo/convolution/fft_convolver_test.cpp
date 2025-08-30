// SPDX-License-Identifier: MIT

#include "fft_convolver.hpp"

#include <neo/algorithm.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/convolution: fft_convolve", "", float, double)
{
    using Float = TestType;

    using namespace neo::convolution;

    auto const empty = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{0};
    auto const one   = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{0};
    REQUIRE(fft_convolve(empty.to_mdspan(), empty.to_mdspan()).extent(0) == 0);
    REQUIRE(fft_convolve(one.to_mdspan(), empty.to_mdspan()).extent(0) == 0);
    REQUIRE(fft_convolve(empty.to_mdspan(), one.to_mdspan()).extent(0) == 0);

    auto const signal_size = GENERATE(as<std::size_t>{}, 8, 9, 10, 64, 78, 143, 256, 444, 666, 1024);
    auto const patch_size  = GENERATE(as<std::size_t>{}, 4, 8, 9, 10, 64, 78, 143, 256, 444, 666, 1024);

    auto const signal = neo::generate_noise_signal<Float>(signal_size, Catch::getSeed());
    auto const patch  = [patch_size] {
        auto buf = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{patch_size};
        buf(0)   = Float(1);
        return buf;
    }();

    auto const output = fft_convolve(signal.to_mdspan(), patch.to_mdspan());
    REQUIRE(output.extent(0) == output_size<mode::full>(signal_size, patch_size));
    REQUIRE(neo::allclose(stdex::submdspan(output.to_mdspan(), std::tuple{0, signal_size}), signal.to_mdspan()));
}
