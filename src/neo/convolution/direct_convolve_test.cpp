// SPDX-License-Identifier: MIT

#include "direct_convolve.hpp"

#include <neo/algorithm.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/convolution: direct_convolve", "", float, double)
{
    using Float = TestType;

    using namespace neo::convolution;

    auto const signal_size = GENERATE(as<std::size_t>{}, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    auto const patch_size  = GENERATE(as<std::size_t>{}, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

    auto const signal = neo::generate_noise_signal<Float>(signal_size, Catch::getSeed());
    auto const patch  = [patch_size] {
        auto buf = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{patch_size};
        buf(0)   = Float(1);
        return buf;
    }();

    auto const output = direct_convolve(signal.to_mdspan(), patch.to_mdspan());
    REQUIRE(output.extent(0) == output_size<mode::full>(signal_size, patch_size));
    REQUIRE(neo::allclose(stdex::submdspan(output.to_mdspan(), std::tuple{0, signal_size}), signal.to_mdspan()));
}
