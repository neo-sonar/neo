// SPDX-License-Identifier: MIT

#include "copy.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/algorithm: copy", "", float, double, std::complex<float>, std::complex<double>)
{
    using Float     = neo::real_or_complex_value_t<TestType>;
    auto const size = GENERATE(as<std::size_t>{}, 2, 33, 128);

    SECTION("vector")
    {
        auto const make_vector = [size](Float val) -> stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> {
            auto vec = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{size};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto out      = make_vector(Float(0));
        auto const in = make_vector(Float(1));
        neo::copy(in.to_mdspan(), out.to_mdspan());
        REQUIRE(neo::allclose(in.to_mdspan(), out.to_mdspan()));
    }

    SECTION("matrix")
    {
        auto const make_matrix = [size](Float val) -> stdex::mdarray<Float, stdex::dextents<std::size_t, 2>> {
            auto vec = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{size, size};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto out      = make_matrix(Float(0));
        auto const in = make_matrix(Float(1));
        neo::copy(in.to_mdspan(), out.to_mdspan());
        REQUIRE(neo::allclose(in.to_mdspan(), out.to_mdspan()));
    }
}
