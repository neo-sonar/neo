// SPDX-License-Identifier: MIT

#include "windowing.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEMPLATE_TEST_CASE("neo/math: rectangular_window", "", float, double)
{
    using Float  = TestType;
    using Window = neo::rectangular_window<Float>;

    auto const size   = GENERATE(as<std::size_t>{}, 128, 256, 512, 1024);
    auto const window = Window{};
    for (auto i{0zu}; i < size; ++i) {
        REQUIRE(window(i, size) == Catch::Approx(Float(1.0)));
    }
}

TEMPLATE_TEST_CASE("neo/math: hann_window", "", float, double)
{
    using Float  = TestType;
    using Window = neo::hann_window<Float>;

    SECTION("compare with numpy.hanning(8)")
    {
        auto const size   = 8;
        auto const window = Window{};
        REQUIRE(window(0, size) == Catch::Approx(Float(0.0)));
        REQUIRE(window(1, size) == Catch::Approx(Float(0.1882551)));
        REQUIRE(window(2, size) == Catch::Approx(Float(0.61126047)));
        REQUIRE(window(3, size) == Catch::Approx(Float(0.95048443)));
        REQUIRE(window(4, size) == Catch::Approx(Float(0.95048443)));
        REQUIRE(window(5, size) == Catch::Approx(Float(0.61126047)));
        REQUIRE(window(6, size) == Catch::Approx(Float(0.1882551)));
        REQUIRE(window(7, size) == Catch::Approx(Float(0.0)));
    }

    SECTION("compare with numpy.hanning(16)")
    {
        auto const size   = 16;
        auto const window = Window{};
        REQUIRE(window(0, size) == Catch::Approx(Float(0.0)));
        REQUIRE(window(1, size) == Catch::Approx(Float(0.04322727)));
        REQUIRE(window(2, size) == Catch::Approx(Float(0.1654347)));
        REQUIRE(window(3, size) == Catch::Approx(Float(0.3454915)));
        REQUIRE(window(4, size) == Catch::Approx(Float(0.55226423)));
        REQUIRE(window(5, size) == Catch::Approx(Float(0.75)));
        REQUIRE(window(6, size) == Catch::Approx(Float(0.9045085)));
        REQUIRE(window(7, size) == Catch::Approx(Float(0.9890738)));
        REQUIRE(window(8, size) == Catch::Approx(Float(0.9890738)));
        REQUIRE(window(9, size) == Catch::Approx(Float(0.9045085)));
        REQUIRE(window(10, size) == Catch::Approx(Float(0.75)));
        REQUIRE(window(11, size) == Catch::Approx(Float(0.55226423)));
        REQUIRE(window(12, size) == Catch::Approx(Float(0.3454915)));
        REQUIRE(window(13, size) == Catch::Approx(Float(0.1654347)));
        REQUIRE(window(14, size) == Catch::Approx(Float(0.04322727)));
        REQUIRE(window(15, size) == Catch::Approx(Float(0.0)));
    }
}

TEMPLATE_TEST_CASE("neo/math: hamming_window", "", float, double)
{
    using Float  = TestType;
    using Window = neo::hamming_window<Float>;

    SECTION("compare with numpy.hamming(8)")
    {
        auto const size   = 8;
        auto const window = Window{};
        REQUIRE(window(0, size) == Catch::Approx(Float(0.08)));
        REQUIRE(window(1, size) == Catch::Approx(Float(0.25319469)));
        REQUIRE(window(2, size) == Catch::Approx(Float(0.64235963)));
        REQUIRE(window(3, size) == Catch::Approx(Float(0.95444568)));
        REQUIRE(window(4, size) == Catch::Approx(Float(0.95444568)));
        REQUIRE(window(5, size) == Catch::Approx(Float(0.64235963)));
        REQUIRE(window(6, size) == Catch::Approx(Float(0.25319469)));
        REQUIRE(window(7, size) == Catch::Approx(Float(0.08)));
    }

    SECTION("compare with numpy.hamming(16)")
    {
        auto const size   = 16;
        auto const window = Window{};
        REQUIRE(window(0, size) == Catch::Approx(Float(0.08)));
        REQUIRE(window(1, size) == Catch::Approx(Float(0.11976909)));
        REQUIRE(window(2, size) == Catch::Approx(Float(0.23219992)));
        REQUIRE(window(3, size) == Catch::Approx(Float(0.39785218)));
        REQUIRE(window(4, size) == Catch::Approx(Float(0.58808309)));
        REQUIRE(window(5, size) == Catch::Approx(Float(0.77)));
        REQUIRE(window(6, size) == Catch::Approx(Float(0.91214782)));
        REQUIRE(window(7, size) == Catch::Approx(Float(0.9899479)));
        REQUIRE(window(8, size) == Catch::Approx(Float(0.9899479)));
        REQUIRE(window(9, size) == Catch::Approx(Float(0.91214782)));
        REQUIRE(window(10, size) == Catch::Approx(Float(0.77)));
        REQUIRE(window(11, size) == Catch::Approx(Float(0.58808309)));
        REQUIRE(window(12, size) == Catch::Approx(Float(0.39785218)));
        REQUIRE(window(13, size) == Catch::Approx(Float(0.23219992)));
        REQUIRE(window(14, size) == Catch::Approx(Float(0.11976909)));
        REQUIRE(window(15, size) == Catch::Approx(Float(0.08)));
    }
}

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/math: generate_window",
    "",
    (neo::rectangular_window, neo::hann_window, neo::hamming_window),
    (float, double)
)
{
    using Window = TestType;
    using Float  = typename Window::real_type;

    auto const size   = GENERATE(as<std::size_t>{}, 128, 256, 512, 1024);
    auto const window = neo::generate_window<Float, Window>(size);
    STATIC_REQUIRE(decltype(window)::rank() == 1);
    REQUIRE(window.extent(0) == size);
}
