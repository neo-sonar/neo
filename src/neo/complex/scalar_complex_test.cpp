// SPDX-License-Identifier: MIT

#include "scalar_complex.hpp"

#include <neo/fixed_point.hpp>
#include <neo/math/conj.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

template<typename TestType>
static auto test_floating_point(TestType tolerance)
{
    using Float   = TestType;
    using Complex = neo::scalar_complex<Float>;

    STATIC_REQUIRE(std::is_trivial_v<Complex>);
    STATIC_REQUIRE(neo::is_complex<neo::scalar_complex<Float>>);
    STATIC_REQUIRE(neo::complex<neo::scalar_complex<Float>>);
    STATIC_REQUIRE(std::same_as<neo::real_or_complex_value_t<neo::scalar_complex<Float>>, Float>);
    STATIC_REQUIRE(std::same_as<typename Complex::value_type, Float>);
    STATIC_REQUIRE(std::tuple_size_v<neo::scalar_complex<Float>> == 2);

    auto tc             = Complex{Float(1)};
    auto const [re, im] = tc;
    REQUIRE(re == Catch::Approx(Float(1)));
    REQUIRE(im == Catch::Approx(Float(0)));

    tc.real(Float(2));
    tc.imag(Float(3));
    REQUIRE(tc.real() == Catch::Approx(Float(2)));
    REQUIRE(tc.imag() == Catch::Approx(Float(3)));

    auto const sum = tc + tc;
    REQUIRE(sum.real() == Catch::Approx(Float(4)));
    REQUIRE(sum.imag() == Catch::Approx(Float(6)));

    auto const diff = tc - sum;
    REQUIRE(diff.real() == Catch::Approx(Float(-2)));
    REQUIRE(diff.imag() == Catch::Approx(Float(-3)));

    auto const product = tc * sum;
    REQUIRE(product.real() == Catch::Approx(Float(-10)));
    REQUIRE(product.imag() == Catch::Approx(Float(24)));

    auto const div_real = product / Float(2);
    REQUIRE(div_real.real() == Catch::Approx(Float(-5)));
    REQUIRE(div_real.imag() == Catch::Approx(Float(12)));

    auto const div = product / Complex{Float(2), Float(4)};
    REQUIRE(div.real() == Catch::Approx(Float(3.8)));
    REQUIRE(div.imag() == Catch::Approx(Float(4.4)));

    auto copy = product;
    copy *= Float(4);
    REQUIRE(copy.real() == Catch::Approx(Float(-40)));
    REQUIRE(copy.imag() == Catch::Approx(Float(96)));

    auto conj = neo::math::conj(copy);
    REQUIRE(conj.real() == Catch::Approx(Float(-40)));
    REQUIRE(conj.imag() == Catch::Approx(Float(-96)));

    auto absolute = neo::abs(copy);
    REQUIRE_THAT(static_cast<double>(absolute), Catch::Matchers::WithinAbs(104, tolerance));
}

template<typename TestType>
static auto test_fixed_point()
{
    using FxPoint = TestType;
    using Complex = neo::scalar_complex<FxPoint>;

    STATIC_REQUIRE(std::is_trivial_v<Complex>);
    STATIC_REQUIRE(neo::is_complex<neo::scalar_complex<FxPoint>>);
    STATIC_REQUIRE(neo::complex<neo::scalar_complex<FxPoint>>);
    STATIC_REQUIRE(std::same_as<neo::real_or_complex_value_t<neo::scalar_complex<FxPoint>>, FxPoint>);
    STATIC_REQUIRE(std::same_as<typename Complex::value_type, FxPoint>);
    STATIC_REQUIRE(std::tuple_size_v<neo::scalar_complex<FxPoint>> == 2);

    auto tc = Complex{FxPoint(neo::underlying_value, 10)};
    REQUIRE(tc.real().value() == 10);
    REQUIRE(tc.imag().value() == 0);

    tc.real(FxPoint(neo::underlying_value, 20));
    tc.imag(FxPoint(neo::underlying_value, 30));
    auto const [re, im] = tc;
    REQUIRE(re.value() == 20);
    REQUIRE(im.value() == 30);

    auto const sum = tc + tc;
    REQUIRE(sum.real().value() == 40);
    REQUIRE(sum.imag().value() == 60);

    auto const diff = tc - sum;
    REQUIRE(diff.real().value() == -20);
    REQUIRE(diff.imag().value() == -30);
}

TEMPLATE_TEST_CASE("neo/complex: scalar_complex", "", float, double)
{
    test_floating_point<TestType>(TestType(0.00001));
}

#if defined(NEO_HAS_BUILTIN_FLOAT16)
TEMPLATE_TEST_CASE("neo/complex: scalar_complex", "", _Float16) { test_floating_point<TestType>(TestType(0.01)); }
#endif

TEMPLATE_TEST_CASE(
    "neo/complex: scalar_complex",
    "",
    neo::q7,
    neo::q15,
    (neo::fixed_point<std::int16_t, 14>),
    (neo::fixed_point<std::int16_t, 13>),
    (neo::fixed_point<std::int16_t, 12>),
    (neo::fixed_point<std::int16_t, 11>),
    (neo::fixed_point<std::int8_t, 6>),
    (neo::fixed_point<std::int8_t, 5>)
)
{
    test_fixed_point<TestType>();
}
