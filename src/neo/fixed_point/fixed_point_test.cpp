// SPDX-License-Identifier: MIT

#include "algorithm.hpp"
#include "complex.hpp"
#include "fixed_point.hpp"
#include "simd.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <functional>
#include <span>
#include <utility>
#include <vector>

[[nodiscard]] static auto approx_equal(float a, float b, float tolerance) -> bool
{
    return std::abs(a - b) < tolerance;
}

// NOLINTBEGIN(-bugprone-branch-clone,-*-braces-around-statements)

template<typename T>
static constexpr auto get_tolerance()
{
    if constexpr (std::same_as<T, neo::q7>) {
        return 0.01F;
    } else if constexpr (std::same_as<T, neo::q15>) {
        return 0.0001F;
    } else if constexpr (std::same_as<typename T::value_type, std::int16_t>) {
        return 0.001F;
    } else {
        return 0.03F;
    }
}

template<typename T>
static constexpr auto tolerance = get_tolerance<T>();

// NOLINTEND(-bugprone-branch-clone,-*-braces-around-statements)

TEMPLATE_TEST_CASE(
    "neo/fixed_point: to_float",
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
    using fxp_t = TestType;

    REQUIRE_THAT(to_float(fxp_t{0.00F}), Catch::Matchers::WithinAbs(0.00F, tolerance<fxp_t>));
    REQUIRE_THAT(to_float(fxp_t{0.12F}), Catch::Matchers::WithinAbs(0.12F, tolerance<fxp_t>));
    REQUIRE_THAT(to_float(fxp_t{0.25F}), Catch::Matchers::WithinAbs(0.25F, tolerance<fxp_t>));
    REQUIRE_THAT(to_float(fxp_t{0.33F}), Catch::Matchers::WithinAbs(0.33F, tolerance<fxp_t>));
    REQUIRE_THAT(to_float(fxp_t{0.40F}), Catch::Matchers::WithinAbs(0.40F, tolerance<fxp_t>));
    REQUIRE_THAT(to_float(fxp_t{0.50F}), Catch::Matchers::WithinAbs(0.50F, tolerance<fxp_t>));
    REQUIRE_THAT(to_float(fxp_t{0.75F}), Catch::Matchers::WithinAbs(0.75F, tolerance<fxp_t>));
}

TEMPLATE_TEST_CASE(
    "neo/fixed_point: to_double",
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
    using fxp_t = TestType;

    REQUIRE_THAT(to_double(fxp_t{0.00}), Catch::Matchers::WithinAbs(0.00, tolerance<fxp_t>));
    REQUIRE_THAT(to_double(fxp_t{0.12}), Catch::Matchers::WithinAbs(0.12, tolerance<fxp_t>));
    REQUIRE_THAT(to_double(fxp_t{0.25}), Catch::Matchers::WithinAbs(0.25, tolerance<fxp_t>));
    REQUIRE_THAT(to_double(fxp_t{0.33}), Catch::Matchers::WithinAbs(0.33, tolerance<fxp_t>));
    REQUIRE_THAT(to_double(fxp_t{0.40}), Catch::Matchers::WithinAbs(0.40, tolerance<fxp_t>));
    REQUIRE_THAT(to_double(fxp_t{0.50}), Catch::Matchers::WithinAbs(0.50, tolerance<fxp_t>));
    REQUIRE_THAT(to_double(fxp_t{0.75}), Catch::Matchers::WithinAbs(0.75, tolerance<fxp_t>));
}

TEMPLATE_TEST_CASE(
    "neo/fixed_point: unary_op",
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
    using fxp_t = TestType;

    auto unary_op = [](auto val, auto op) {
        auto const result   = to_float(op(fxp_t{val}));
        auto const expected = op(val);
        return approx_equal(result, expected, tolerance<fxp_t>);
    };

    // operator+
    auto const unary_plus = [](auto val) { return +val; };
    REQUIRE(unary_op(0.00F, unary_plus));
    REQUIRE(unary_op(0.12F, unary_plus));
    REQUIRE(unary_op(0.25F, unary_plus));
    REQUIRE(unary_op(0.33F, unary_plus));
    REQUIRE(unary_op(0.40F, unary_plus));
    REQUIRE(unary_op(0.45F, unary_plus));
    REQUIRE(unary_op(0.49F, unary_plus));
    REQUIRE(unary_op(0.75F, unary_plus));
    REQUIRE(unary_op(0.99F, unary_plus));

    REQUIRE(unary_op(-0.00F, unary_plus));
    REQUIRE(unary_op(-0.12F, unary_plus));
    REQUIRE(unary_op(-0.25F, unary_plus));
    REQUIRE(unary_op(-0.33F, unary_plus));
    REQUIRE(unary_op(-0.40F, unary_plus));
    REQUIRE(unary_op(-0.45F, unary_plus));
    REQUIRE(unary_op(-0.49F, unary_plus));
    REQUIRE(unary_op(-0.75F, unary_plus));
    REQUIRE(unary_op(-0.99F, unary_plus));

    // operator-
    REQUIRE(unary_op(0.00F, std::negate()));
    REQUIRE(unary_op(0.12F, std::negate()));
    REQUIRE(unary_op(0.25F, std::negate()));
    REQUIRE(unary_op(0.33F, std::negate()));
    REQUIRE(unary_op(0.40F, std::negate()));
    REQUIRE(unary_op(0.45F, std::negate()));
    REQUIRE(unary_op(0.49F, std::negate()));
    REQUIRE(unary_op(0.75F, std::negate()));
    REQUIRE(unary_op(0.99F, std::negate()));

    REQUIRE(unary_op(-0.00F, std::negate()));
    REQUIRE(unary_op(-0.12F, std::negate()));
    REQUIRE(unary_op(-0.25F, std::negate()));
    REQUIRE(unary_op(-0.33F, std::negate()));
    REQUIRE(unary_op(-0.40F, std::negate()));
    REQUIRE(unary_op(-0.45F, std::negate()));
    REQUIRE(unary_op(-0.49F, std::negate()));
    REQUIRE(unary_op(-0.75F, std::negate()));
    REQUIRE(unary_op(-0.99F, std::negate()));
}

TEMPLATE_TEST_CASE(
    "neo/fixed_point: binary_op",
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
    using fxp_t = TestType;

    auto binary_op = [](auto lhs, auto rhs, auto op) {
        auto const result   = to_float(op(fxp_t{lhs}, fxp_t{rhs}));
        auto const expected = op(lhs, rhs);
        return approx_equal(result, expected, tolerance<fxp_t>);
    };

    // operator+
    REQUIRE(binary_op(0.00F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.12F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.25F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.33F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.40F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.45F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.49F, 0.5F, std::plus()));

    // operator-
    REQUIRE(binary_op(0.00F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.12F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.25F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.33F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.40F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.45F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.49F, 0.5F, std::minus()));

    // operator*
    REQUIRE(binary_op(0.00F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.12F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.25F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.33F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.40F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.45F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.49F, 0.5F, std::multiplies()));
}

TEMPLATE_TEST_CASE(
    "neo/fixed_point: comparison",
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
    using fxp_t = TestType;

    auto compare_op = [](auto lhs, auto rhs, auto op) {
        auto const result   = op(fxp_t{lhs}, fxp_t{rhs});
        auto const expected = op(lhs, rhs);
        return result == expected;
    };

    // operator==
    REQUIRE(compare_op(+0.00F, +0.00F, std::equal_to()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::equal_to()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::equal_to()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::equal_to()));

    // operator!=
    REQUIRE(compare_op(+0.00F, +0.00F, std::not_equal_to()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::not_equal_to()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::not_equal_to()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::not_equal_to()));

    // operator<
    REQUIRE(compare_op(+0.00F, +0.00F, std::less()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::less()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::less()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::less()));

    // operator<=
    REQUIRE(compare_op(+0.00F, +0.00F, std::less_equal()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::less_equal()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::less_equal()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::less_equal()));

    // operator>
    REQUIRE(compare_op(+0.00F, +0.00F, std::greater()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::greater()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::greater()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::greater()));

    // operator>=
    REQUIRE(compare_op(+0.00F, +0.00F, std::greater_equal()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::greater_equal()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::greater_equal()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::greater_equal()));
}

TEMPLATE_TEST_CASE(
    "neo/fixed_point: add",
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
    using fxp_t = TestType;

    SECTION("empty")
    {
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    SECTION("scalar")
    {
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F + 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }

    SECTION("vectorized")
    {
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F + 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }
}

TEMPLATE_TEST_CASE(
    "neo/fixed_point: subtract",
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
    using fxp_t = TestType;

    SECTION("empty")
    {
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    SECTION("scalar")
    {
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F - 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }

    SECTION("vectorized")
    {
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F - 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }
}

TEMPLATE_TEST_CASE(
    "neo/fixed_point: multiply",
    "",
    neo::q15,
    (neo::fixed_point<std::int16_t, 14>),
    (neo::fixed_point<std::int16_t, 13>),
    (neo::fixed_point<std::int16_t, 12>),
    (neo::fixed_point<std::int16_t, 11>),
    neo::q7,
    (neo::fixed_point<std::int8_t, 6>),
    (neo::fixed_point<std::int8_t, 5>)
)
{
    using fxp_t = TestType;

    SECTION("empty")
    {
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    SECTION("scalar")
    {
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F * 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }

    SECTION("vectorized")
    {
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F * 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }
}

TEST_CASE("neo/fixed_point: complex_q7")
{
    STATIC_REQUIRE(neo::complex<neo::complex_q7>);
    STATIC_REQUIRE(neo::is_complex<neo::complex_q7>);

    auto const lhs = neo::complex_q7{
        neo::q7{neo::underlying_value, 14},
        neo::q7{neo::underlying_value, 42}
    };
    auto const rhs = neo::complex_q7{
        neo::q7{neo::underlying_value, 10},
        neo::q7{neo::underlying_value, 20}
    };

    auto const sum = lhs + rhs;
    REQUIRE(static_cast<int>(sum.real().value()) == 24);
    REQUIRE(static_cast<int>(sum.imag().value()) == 62);

    auto const diff = lhs - rhs;
    REQUIRE(static_cast<int>(diff.real().value()) == 4);
    REQUIRE(static_cast<int>(diff.imag().value()) == 22);
}

TEST_CASE("neo/fixed_point: complex_q15")
{
    STATIC_REQUIRE(neo::complex<neo::complex_q15>);
    STATIC_REQUIRE(neo::is_complex<neo::complex_q15>);

    auto const lhs = neo::complex_q15{
        neo::q15{neo::underlying_value, 1430},
        neo::q15{neo::underlying_value,  420}
    };
    auto const rhs = neo::complex_q15{
        neo::q15{neo::underlying_value, 1000},
        neo::q15{neo::underlying_value,  100}
    };

    auto const sum = lhs + rhs;
    REQUIRE(sum.real().value() == 2430);
    REQUIRE(sum.imag().value() == 520);

    auto const diff = lhs - rhs;
    REQUIRE(diff.real().value() == 430);
    REQUIRE(diff.imag().value() == 320);

    auto const product = lhs * rhs;
    REQUIRE(product.real().value() == 9);
    REQUIRE(product.imag().value() == 7);
}

template<typename Batch>
static auto test_simd_fixed_point()
{
    using Scalar = typename Batch::value_type;
    using Int    = typename Scalar::storage_type;

    auto lhs = std::array<Scalar, Batch::size>{};
    auto rhs = std::array<Scalar, Batch::size>{};
    auto out = std::array<Scalar, Batch::size>{};

    std::fill(lhs.begin(), lhs.end(), Scalar{neo::underlying_value, Int(50)});
    std::fill(rhs.begin(), rhs.end(), Scalar{neo::underlying_value, Int(42)});
    auto const l = Batch::load_unaligned(lhs.data());
    auto const r = Batch::load_unaligned(rhs.data());

    auto const sum = l + r;
    sum.store_unaligned(out.data());
    for (auto const val : out) {
        REQUIRE(val.value() == Int(92));
    }

    auto const diff = l - r;
    diff.store_unaligned(out.data());
    for (auto const val : out) {
        REQUIRE(val.value() == Int(8));
    }

    auto const brodcasted = Batch::broadcast(Scalar{neo::underlying_value, Int(123)});
    brodcasted.store_unaligned(out.data());
    for (auto const val : out) {
        REQUIRE(val.value() == Int(123));
    }
}

#if defined(NEO_HAS_ISA_SSE2) or defined(NEO_HAS_ISA_NEON)

TEMPLATE_TEST_CASE("neo/fixed_point: batch", "", neo::q7x16, neo::q15x8)
{
    STATIC_REQUIRE(TestType::alignment == 16);
    STATIC_REQUIRE(TestType::size == 16 / sizeof(typename TestType::value_type));
    test_simd_fixed_point<TestType>();
}

#endif

#if defined(NEO_HAS_ISA_AVX2)

TEMPLATE_TEST_CASE("neo/fixed_point: batch", "", neo::q7x32, neo::q15x16)
{
    STATIC_REQUIRE(TestType::alignment == 32);
    STATIC_REQUIRE(TestType::size == 32 / sizeof(typename TestType::value_type));
    test_simd_fixed_point<TestType>();
}

#endif

#if defined(NEO_HAS_ISA_AVX512BW)

TEMPLATE_TEST_CASE("neo/fixed_point: batch", "", neo::q7x64, neo::q15x32)
{
    STATIC_REQUIRE(TestType::alignment == 64);
    STATIC_REQUIRE(TestType::size == 64 / sizeof(typename TestType::value_type));
    test_simd_fixed_point<TestType>();
}

#endif
