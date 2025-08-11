// SPDX-License-Identifier: MIT

#include "imag.hpp"

#include <neo/complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE(
    "neo/math: imag",
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

    REQUIRE(neo::math::imag(Float{0}) == Catch::Approx(0.0));

    if (neo::complex<Float>) {
        REQUIRE(neo::math::imag(Float{1}) == Catch::Approx(0.0));
    } else {
        REQUIRE(neo::math::imag(Float{1}) == Catch::Approx(1.0));
    }
}
