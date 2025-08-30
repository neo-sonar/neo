// SPDX-License-Identifier: MIT

#include "mean.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/fill.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEMPLATE_TEST_CASE("neo/algorithm: mean", "", float, double)
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

        auto const empty = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{};
        auto const zeros = make_vector(Float(0));
        auto const ones  = make_vector(Float(1));

        REQUIRE_FALSE(neo::mean(empty.to_mdspan()).has_value());

        REQUIRE(neo::mean(zeros.to_mdspan()).has_value());
        REQUIRE_THAT(neo::mean(zeros.to_mdspan()).value(), Catch::Matchers::WithinAbs(0.0, 0.000001));

        REQUIRE(neo::mean(ones.to_mdspan()).has_value());
        REQUIRE_THAT(neo::mean(ones.to_mdspan()).value(), Catch::Matchers::WithinAbs(1.0, 0.000001));

        auto const vals   = std::array{Float(1), Float(2), Float(3)};
        auto const values = stdex::mdspan{vals.data(), stdex::extents{vals.size()}};
        REQUIRE(neo::mean(values).has_value());
        REQUIRE_THAT(neo::mean(values).value(), Catch::Matchers::WithinAbs(2.0, 0.000001));
    }

    SECTION("matrix")
    {
        auto const make_matrix = [size](Float val) -> stdex::mdarray<Float, stdex::dextents<std::size_t, 2>> {
            auto vec = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{size, size * std::size_t(2)};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto const empty = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{};
        auto const zeros = make_matrix(Float(0));
        auto const ones  = make_matrix(Float(1));

        REQUIRE_FALSE(neo::mean(empty.to_mdspan()).has_value());

        REQUIRE(neo::mean(zeros.to_mdspan()).has_value());
        REQUIRE_THAT(neo::mean(zeros.to_mdspan()).value(), Catch::Matchers::WithinAbs(0.0, 0.000001));

        REQUIRE(neo::mean(ones.to_mdspan()).has_value());
        REQUIRE_THAT(neo::mean(ones.to_mdspan()).value(), Catch::Matchers::WithinAbs(1.0, 0.000001));

        auto const vals   = std::array{Float(2), Float(1), Float(2), Float(3)};
        auto const values = stdex::mdspan{vals.data(), stdex::extents(2, 2)};
        REQUIRE(neo::mean(values).has_value());
        REQUIRE_THAT(neo::mean(values).value(), Catch::Matchers::WithinAbs(2.0, 0.000001));
    }
}
