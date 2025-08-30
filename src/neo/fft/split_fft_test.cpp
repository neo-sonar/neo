// SPDX-License-Identifier: MIT

#include "split_fft.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex/scalar_complex.hpp>
#include <neo/fft.hpp>
#include <neo/simd.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <random>
#include <vector>

namespace {

template<typename Plan>
auto test_split_fft_plan() -> void
{
    using Float = typename Plan::value_type;

    auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    CAPTURE(order);

    auto plan = Plan{neo::fft::from_order, order};
    REQUIRE(plan.order() == order);
    REQUIRE(plan.size() == neo::ipow<2zu>(order));

    auto const noise = neo::generate_noise_signal<Float>(plan.size(), Catch::getSeed());

    SECTION("inplace")
    {
        auto buf = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{2, noise.extent(0)};
        auto z   = neo::split_complex{
            stdex::submdspan(buf.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(buf.to_mdspan(), 1, stdex::full_extent),
        };

        neo::copy(noise.to_mdspan(), z.real);
        neo::fft::fft(plan, z);
        neo::fft::ifft(plan, z);

        neo::scale(Float(1) / static_cast<Float>(plan.size()), z.real);
        REQUIRE(neo::allclose(noise.to_mdspan(), z.real));
    }

    SECTION("copy")
    {
        auto in_buf  = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{2, noise.extent(0)};
        auto out_buf = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{2, noise.extent(0)};
        auto in_z    = neo::split_complex{
            stdex::submdspan(in_buf.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(in_buf.to_mdspan(), 1, stdex::full_extent),
        };
        auto out_z = neo::split_complex{
            stdex::submdspan(out_buf.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(out_buf.to_mdspan(), 1, stdex::full_extent),
        };

        neo::copy(noise.to_mdspan(), in_z.real);
        neo::fft::fft(plan, in_z, out_z);
        neo::fft::ifft(plan, out_z, in_z);

        neo::scale(Float(1) / static_cast<Float>(plan.size()), in_z.real);
        REQUIRE(neo::allclose(noise.to_mdspan(), in_z.real));
    }
}

}  // namespace

#if defined(NEO_HAS_INTEL_IPP)
TEMPLATE_TEST_CASE("neo/fft: intel_ipp_split_fft_plan", "", float, double)
{
    test_split_fft_plan<neo::fft::intel_ipp_split_fft_plan<TestType>>();
}
#endif

#if defined(NEO_HAS_APPLE_ACCELERATE)
TEMPLATE_TEST_CASE("neo/fft: apple_vdsp_split_fft_plan", "", float, double)
{
    test_split_fft_plan<neo::fft::apple_vdsp_split_fft_plan<TestType>>();
}
#endif

TEMPLATE_TEST_CASE("neo/fft: fallback_split_fft_plan", "", float, double)
{
    test_split_fft_plan<neo::fft::fallback_split_fft_plan<TestType>>();
}

TEMPLATE_TEST_CASE("neo/fft: split_fft_plan", "", float, double)
{
    test_split_fft_plan<neo::fft::split_fft_plan<TestType>>();
}
