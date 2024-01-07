// SPDX-License-Identifier: MIT

#include "bit_log2.hpp"

#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

TEMPLATE_TEST_CASE("neo/bit: bit_log2", "", std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t)
{
    using Int = TestType;

    STATIC_REQUIRE(std::same_as<decltype(neo::bit_log2(std::declval<Int>())), Int>);

    REQUIRE(neo::bit_log2(Int(1)) == Int(0));
    REQUIRE(neo::bit_log2(Int(2)) == Int(1));
    REQUIRE(neo::bit_log2(Int(4)) == Int(2));
    REQUIRE(neo::bit_log2(Int(8)) == Int(3));
    REQUIRE(neo::bit_log2(Int(16)) == Int(4));
    REQUIRE(neo::bit_log2(Int(32)) == Int(5));
    REQUIRE(neo::bit_log2(Int(64)) == Int(6));

    if constexpr (sizeof(Int) > 1) {
        REQUIRE(neo::bit_log2(Int(128)) == Int(7));
        REQUIRE(neo::bit_log2(Int(256)) == Int(8));
        REQUIRE(neo::bit_log2(Int(512)) == Int(9));
        REQUIRE(neo::bit_log2(Int(1024)) == Int(10));
        REQUIRE(neo::bit_log2(Int(2048)) == Int(11));
        REQUIRE(neo::bit_log2(Int(4096)) == Int(12));
        REQUIRE(neo::bit_log2(Int(8192)) == Int(13));
    }
}
