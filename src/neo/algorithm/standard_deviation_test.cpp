// SPDX-License-Identifier: MIT

#include "standard_deviation.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/fill.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEMPLATE_TEST_CASE("neo/algorithm: standard_deviation", "", float, double)
{
    using Float = TestType;

    auto const size = GENERATE(as<std::size_t>{}, 2, 33, 128);

    SECTION("vector")
    {
        auto const make_vector = [size](Float val) -> stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> {
            auto vec = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{size};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto const size_0 = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{};
        auto const size_1 = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{1};
        auto const zeros  = make_vector(Float(0));
        auto const ones   = make_vector(Float(1));

        REQUIRE_FALSE(neo::standard_deviation(size_0.to_mdspan()).has_value());
        REQUIRE_FALSE(neo::standard_deviation(size_1.to_mdspan()).has_value());

        REQUIRE(neo::standard_deviation(zeros.to_mdspan()).has_value());
        REQUIRE_THAT(neo::standard_deviation(zeros.to_mdspan()).value(), Catch::Matchers::WithinAbs(0.0, 0.000001));

        REQUIRE(neo::standard_deviation(ones.to_mdspan()).has_value());
        REQUIRE_THAT(neo::standard_deviation(ones.to_mdspan()).value(), Catch::Matchers::WithinAbs(0.0, 0.000001));

        auto const vals   = std::array{Float(2), Float(4), Float(4), Float(4), Float(5), Float(5), Float(7), Float(9)};
        auto const values = stdex::mdspan{vals.data(), stdex::extents{vals.size()}};
        REQUIRE(neo::standard_deviation(values).has_value());
        REQUIRE_THAT(neo::standard_deviation(values).value(), Catch::Matchers::WithinAbs(2.0, 0.000001));
    }

    SECTION("matrix")
    {
        auto const make_matrix = [size](Float val) -> stdex::mdarray<Float, stdex::dextents<std::size_t, 2>> {
            auto vec = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{size, size * std::size_t(2)};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto const size_0 = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{};
        auto const size_1 = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{1, 1};
        auto const zeros  = make_matrix(Float(0));
        auto const ones   = make_matrix(Float(1));

        REQUIRE_FALSE(neo::standard_deviation(size_0.to_mdspan()).has_value());
        REQUIRE_FALSE(neo::standard_deviation(size_1.to_mdspan()).has_value());

        REQUIRE(neo::standard_deviation(zeros.to_mdspan()).has_value());
        REQUIRE_THAT(neo::standard_deviation(zeros.to_mdspan()).value(), Catch::Matchers::WithinAbs(0.0, 0.000001));

        REQUIRE(neo::standard_deviation(ones.to_mdspan()).has_value());
        REQUIRE_THAT(neo::standard_deviation(ones.to_mdspan()).value(), Catch::Matchers::WithinAbs(0.0, 0.000001));

        auto const vals   = std::array{Float(2), Float(4), Float(4), Float(4), Float(5), Float(5), Float(7), Float(9)};
        auto const values = stdex::mdspan{vals.data(), stdex::extents(2, 4)};
        REQUIRE(neo::standard_deviation(values).has_value());
        REQUIRE_THAT(neo::standard_deviation(values).value(), Catch::Matchers::WithinAbs(2.0, 0.000001));
    }
}
