// SPDX-License-Identifier: MIT

#include "abs.hpp"
#include "conj.hpp"

#include <neo/complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE(
    "neo/math: abs",
    "",
    std::int8_t,
    std::uint8_t,
    std::int16_t,
    std::uint16_t,
    std::int32_t,
    std::uint32_t,
    std::int64_t,
    std::uint64_t
)
{
    using Int = TestType;

    REQUIRE(neo::math::abs(Int(0)) == Int(0));
    REQUIRE(neo::math::abs(Int(1)) == Int(1));
    REQUIRE(neo::math::abs(Int(2)) == Int(2));
    REQUIRE(neo::math::abs(Int(3)) == Int(3));

    if (std::signed_integral<Int>) {
        REQUIRE(neo::math::abs(Int(-1)) == Int(1));
        REQUIRE(neo::math::abs(Int(-2)) == Int(2));
        REQUIRE(neo::math::abs(Int(-3)) == Int(3));
    }
}

TEMPLATE_TEST_CASE("neo/math: abs", "", float, double)
{
    using Float = TestType;

    REQUIRE(neo::math::abs(Float(0)) == Catch::Approx(Float(0)));
    REQUIRE(neo::math::abs(Float(1)) == Catch::Approx(Float(1)));
    REQUIRE(neo::math::abs(Float(1.25)) == Catch::Approx(Float(1.25)));

    REQUIRE(neo::math::abs(Float(-0)) == Catch::Approx(Float(0)));
    REQUIRE(neo::math::abs(Float(-1)) == Catch::Approx(Float(1)));
    REQUIRE(neo::math::abs(Float(-1.25)) == Catch::Approx(Float(1.25)));
}

TEMPLATE_TEST_CASE("neo/math: abs", "", neo::complex64, neo::complex128)
{
    using Complex = TestType;

    REQUIRE(neo::math::abs(Complex{0, 0}) == Catch::Approx(0));
    REQUIRE(neo::math::abs(Complex{1, 0}) == Catch::Approx(1));
    REQUIRE(neo::math::conj(Complex{1, 2}).real() == Catch::Approx(1));
    REQUIRE(neo::math::conj(Complex{1, 2}).imag() == Catch::Approx(-2));
}

namespace ns {

struct complex
{
    double x;

    auto abs() const noexcept { return x; }

    auto conj() const noexcept { return x; }
};

}  // namespace ns

TEMPLATE_TEST_CASE("neo/math: abs", "", ns::complex)
{
    using Complex = TestType;

    REQUIRE(neo::math::abs(Complex{0.0}) == Catch::Approx(0.0));
    REQUIRE(neo::math::abs(Complex{1.0}) == Catch::Approx(1.0));
    REQUIRE(neo::math::conj(Complex{1.0}) == Catch::Approx(1.0));
}
