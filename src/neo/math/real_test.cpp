// SPDX-License-Identifier: MIT

#include "real.hpp"

#include <neo/complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE(
    "neo/math: real",
    "",
    float,
    double,
    neo::complex64,
    neo::complex128,
    std::complex<float>,
    std::complex<double>
)
{
    using Float = TestType;

    REQUIRE(neo::math::real(Float{0}) == Catch::Approx(0.0));
    REQUIRE(neo::math::real(Float{1}) == Catch::Approx(1.0));
}
