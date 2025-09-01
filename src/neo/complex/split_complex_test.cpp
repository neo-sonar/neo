// SPDX-License-Identifier: MIT

#include "split_complex.hpp"

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/complex: split_complex", "", float, double)
{
    using Float = TestType;

    auto const size = std::size_t(42);

    auto buffer = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
    auto split  = neo::split_complex{
        stdex::submdspan(buffer.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(buffer.to_mdspan(), 1, stdex::full_extent),
    };

    REQUIRE(split.real.extent(0) == 42zu);
    REQUIRE(split.imag.extent(0) == 42zu);
}
