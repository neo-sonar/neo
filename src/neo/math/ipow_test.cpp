// SPDX-License-Identifier: MIT

#include "ipow.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE(
    "neo/math: ipow",
    "",
    std::int16_t,
    std::uint16_t,
    std::int32_t,
    std::uint32_t,
    std::int64_t,
    std::uint64_t
)
{
    using Int = TestType;

    REQUIRE(neo::ipow<Int(1)>(Int(0)) == Int(1));
    REQUIRE(neo::ipow<Int(1)>(Int(1)) == Int(1));
    REQUIRE(neo::ipow<Int(1)>(Int(2)) == Int(1));
    REQUIRE(neo::ipow<Int(1)>(Int(3)) == Int(1));

    REQUIRE(neo::ipow<Int(2)>(Int(0)) == Int(1));
    REQUIRE(neo::ipow<Int(2)>(Int(1)) == Int(2));
    REQUIRE(neo::ipow<Int(2)>(Int(2)) == Int(4));
    REQUIRE(neo::ipow<Int(2)>(Int(3)) == Int(8));

    REQUIRE(neo::ipow<Int(3)>(Int(0)) == Int(1));
    REQUIRE(neo::ipow<Int(3)>(Int(1)) == Int(3));
    REQUIRE(neo::ipow<Int(3)>(Int(2)) == Int(9));
    REQUIRE(neo::ipow<Int(3)>(Int(3)) == Int(27));

    REQUIRE(neo::ipow<Int(4)>(Int(0)) == Int(1));
    REQUIRE(neo::ipow<Int(4)>(Int(1)) == Int(4));
    REQUIRE(neo::ipow<Int(4)>(Int(2)) == Int(16));
    REQUIRE(neo::ipow<Int(4)>(Int(3)) == Int(64));

    REQUIRE(neo::ipow<Int(8)>(Int(0)) == Int(1));
    REQUIRE(neo::ipow<Int(8)>(Int(1)) == Int(8));
    REQUIRE(neo::ipow<Int(8)>(Int(2)) == Int(64));
    REQUIRE(neo::ipow<Int(8)>(Int(3)) == Int(512));

    REQUIRE(neo::ipow<Int(10)>(Int(0)) == Int(1));
    REQUIRE(neo::ipow<Int(10)>(Int(1)) == Int(10));
    REQUIRE(neo::ipow<Int(10)>(Int(2)) == Int(100));

    REQUIRE(neo::ipow<Int(16)>(Int(0)) == Int(1));
    REQUIRE(neo::ipow<Int(16)>(Int(1)) == Int(16));
    REQUIRE(neo::ipow<Int(16)>(Int(2)) == Int(256));
    REQUIRE(neo::ipow<Int(16)>(Int(3)) == Int(4096));

    REQUIRE(neo::ipow<Int(32)>(Int(0)) == Int(1));
    REQUIRE(neo::ipow<Int(32)>(Int(1)) == Int(32));
    REQUIRE(neo::ipow<Int(32)>(Int(2)) == Int(1024));
}
