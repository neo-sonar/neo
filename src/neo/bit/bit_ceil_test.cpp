// SPDX-License-Identifier: MIT

#include "bit_ceil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <random>

TEMPLATE_TEST_CASE("neo/bit: bit_ceil", "", std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t)
{
    using UInt = TestType;

    REQUIRE(neo::bit_ceil(UInt(1)) == UInt(1));
    REQUIRE(neo::bit_ceil(UInt(2)) == UInt(2));
    REQUIRE(neo::bit_ceil(UInt(3)) == UInt(4));
    REQUIRE(neo::bit_ceil(UInt(4)) == UInt(4));
    REQUIRE(neo::bit_ceil(UInt(100)) == UInt(128));
}

TEMPLATE_TEST_CASE("neo/bit: detail::bit_ceil_fallback", "", std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t)
{
    using UInt = TestType;

    REQUIRE(neo::detail::bit_ceil_fallback(UInt(1)) == UInt(1));
    REQUIRE(neo::detail::bit_ceil_fallback(UInt(2)) == UInt(2));
    REQUIRE(neo::detail::bit_ceil_fallback(UInt(3)) == UInt(4));
    REQUIRE(neo::detail::bit_ceil_fallback(UInt(4)) == UInt(4));
    REQUIRE(neo::detail::bit_ceil_fallback(UInt(100)) == UInt(128));

#if defined(__cpp_lib_int_pow2)
    // Can't use std::uniform_int_distribution<UInt> directly,
    // wouldn't compile for "unsigned char" on Microsoft STL.
    // Fails static assertion. Workaround: always use std::uint64_t
    // and cast to UInt afterwards.

    auto rng  = std::mt19937{Catch::getSeed()};
    auto dist = std::uniform_int_distribution<std::uint64_t>{1, std::numeric_limits<UInt>::max() / 2};

    for (auto i{0}; i < 10'000; ++i) {
        auto const val = static_cast<UInt>(dist(rng));
        CAPTURE(val);
        REQUIRE(neo::detail::bit_ceil_fallback(val) == std::bit_ceil(val));
    }

#endif
}
