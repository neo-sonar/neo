// SPDX-License-Identifier: MIT

#include "mean_squared_error.hpp"

#include <neo/algorithm/fill.hpp>
#include <neo/complex/scalar_complex.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE(
    "neo/algorithm: mean_squared_error",
    "",
    float,
    double,
    std::complex<float>,
    std::complex<double>,
    neo::complex64,
    neo::complex128
)
{
    using Scalar = TestType;
    using Float  = neo::real_or_complex_value_t<Scalar>;

    auto const size        = GENERATE(as<std::size_t>{}, 2, 33, 128);
    auto const fill_vector = [size](Scalar val) -> stdex::mdarray<Scalar, stdex::dextents<std::size_t, 1>> {
        auto vec = stdex::mdarray<Scalar, stdex::dextents<std::size_t, 1>>{size};
        neo::fill(vec.to_mdspan(), val);
        return vec;
    };
    auto const fill_matrix = [size](Scalar val) -> stdex::mdarray<Scalar, stdex::dextents<std::size_t, 2>> {
        auto vec = stdex::mdarray<Scalar, stdex::dextents<std::size_t, 2>>{size, size * 2};
        neo::fill(vec.to_mdspan(), val);
        return vec;
    };

    SECTION("vector")
    {
        auto const lhs = fill_vector(Scalar(Float(0)));
        auto const rhs = fill_vector(Scalar(Float(1)));
        REQUIRE(neo::mean_squared_error(lhs.to_mdspan(), rhs.to_mdspan()) == Catch::Approx(1.0));
    }

    SECTION("matrix")
    {
        auto const lhs = fill_matrix(Scalar(Float(0)));
        auto const rhs = fill_matrix(Scalar(Float(1)));
        REQUIRE(neo::mean_squared_error(lhs.to_mdspan(), rhs.to_mdspan()) == Catch::Approx(1.0));
    }
}
