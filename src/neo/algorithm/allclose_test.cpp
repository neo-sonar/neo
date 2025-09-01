// SPDX-License-Identifier: MIT

#include "allclose.hpp"

#include <neo/algorithm/fill.hpp>
#include <neo/complex/scalar_complex.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE(
    "neo/algorithm: allclose",
    "",
    float,
    double,
    std::complex<float>,
    std::complex<double>,
    neo::complex64,
    neo::complex128
)
{
    using Float = neo::real_or_complex_value_t<TestType>;

    auto const size = GENERATE(as<std::size_t>{}, 2, 33, 128);

    SECTION("vector")
    {
        auto const make_vector = [size](Float val) -> stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> {
            auto vec = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{size};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto const zeros       = make_vector(Float(0));
        auto const ones        = make_vector(Float(1));
        auto const almost_ones = make_vector(Float(0.95));

        REQUIRE_FALSE(neo::allclose(ones.to_mdspan(), zeros.to_mdspan()));
        REQUIRE_FALSE(neo::allclose(ones.to_mdspan(), almost_ones.to_mdspan()));

        REQUIRE(neo::allclose(zeros.to_mdspan(), zeros.to_mdspan()));
        REQUIRE(neo::allclose(ones.to_mdspan(), ones.to_mdspan()));
        REQUIRE(neo::allclose(ones.to_mdspan(), almost_ones.to_mdspan(), Float(0.7)));
    }

    SECTION("matrix")
    {
        auto const make_matrix = [size](Float val) -> stdex::mdarray<Float, stdex::dextents<std::size_t, 2>> {
            auto vec = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{size, size * 2zu};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto const zeros       = make_matrix(Float(0));
        auto const ones        = make_matrix(Float(1));
        auto const almost_ones = make_matrix(Float(0.95));

        REQUIRE_FALSE(neo::allclose(ones.to_mdspan(), zeros.to_mdspan()));
        REQUIRE_FALSE(neo::allclose(ones.to_mdspan(), almost_ones.to_mdspan()));

        REQUIRE(neo::allclose(zeros.to_mdspan(), zeros.to_mdspan()));
        REQUIRE(neo::allclose(ones.to_mdspan(), ones.to_mdspan()));
        REQUIRE(neo::allclose(ones.to_mdspan(), almost_ones.to_mdspan(), Float(0.7)));
    }
}
