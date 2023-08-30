#include "energy_normalize.hpp"

#include <neo/algorithm/fill.hpp>
#include <neo/complex/scalar_complex.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/algorithm: energy_normalize", "", float, double)
{
    using Float = TestType;

    auto const size        = GENERATE(as<std::size_t>{}, 2, 33, 128);
    auto const set1_vector = [size](Float val) {
        auto vec = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{size};
        vec(0)   = val;
        return vec;
    };
    auto const set1_matrix = [size](Float val) {
        auto mat  = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{size, size * 2};
        mat(0, 0) = val;
        return mat;
    };

    SECTION("vector")
    {
        REQUIRE(neo::energy_normalization_factor(set1_vector(Float(0)).to_mdspan()) == Catch::Approx(1.0));
        REQUIRE(neo::energy_normalization_factor(set1_vector(Float(1)).to_mdspan()) == Catch::Approx(1.0));
        REQUIRE(neo::energy_normalization_factor(set1_vector(Float(2)).to_mdspan()) == Catch::Approx(0.5));

        auto vec = set1_vector(Float(2));
        neo::energy_normalize(vec.to_mdspan());
        REQUIRE(vec(0) == Catch::Approx(1.0));

        vec(0) = Float(2);
        vec(1) = Float(2);
        REQUIRE(neo::energy_normalization_factor(vec.to_mdspan()) == Catch::Approx(0.353553391));
        neo::energy_normalize(vec.to_mdspan());
        REQUIRE(vec(0) == Catch::Approx(0.707106782));
        REQUIRE(vec(1) == Catch::Approx(0.707106782));
    }

    SECTION("matrix")
    {
        REQUIRE(neo::energy_normalization_factor(set1_matrix(Float(0)).to_mdspan()) == Catch::Approx(1.0));
        REQUIRE(neo::energy_normalization_factor(set1_matrix(Float(1)).to_mdspan()) == Catch::Approx(1.0));
        REQUIRE(neo::energy_normalization_factor(set1_matrix(Float(2)).to_mdspan()) == Catch::Approx(0.5));

        auto mat = set1_matrix(Float(2));
        neo::energy_normalize(mat.to_mdspan());
        REQUIRE(mat(0, 0) == Catch::Approx(1.0));

        mat(0, 0) = Float(2);
        mat(0, 1) = Float(2);
        REQUIRE(neo::energy_normalization_factor(mat.to_mdspan()) == Catch::Approx(0.353553391));
        neo::energy_normalize(mat.to_mdspan());
        REQUIRE(mat(0, 0) == Catch::Approx(0.707106782));
        REQUIRE(mat(0, 1) == Catch::Approx(0.707106782));
    }
}
