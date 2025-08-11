// SPDX-License-Identifier: MIT

#include "multiply_add.hpp"

#include <neo/algorithm/add.hpp>
#include <neo/algorithm/allmatch.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/math/float_equality.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

template<typename Float>
static auto test_csr_matrix()
{
    auto is_zero = [](auto x) { return neo::float_equality::exact(x, Float(0)); };

    auto lhs = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{16, 32};
    std::fill(lhs.data(), std::next(lhs.data(), std::ssize(lhs)), Float(1));

    auto rhs = neo::csr_matrix<Float>{16, 32};
    REQUIRE(rhs.rows() == 16);
    REQUIRE(rhs.columns() == 32);

    auto accumulator = std::vector<Float>(lhs.extent(1));
    auto acc         = stdex::mdspan{accumulator.data(), stdex::extents{accumulator.size()}};
    auto left_row0   = stdex::submdspan(lhs.to_mdspan(), 0, stdex::full_extent);

    neo::multiply_add(left_row0, rhs, 0, acc, acc);
    REQUIRE(neo::allmatch(acc, is_zero));

    rhs.insert(0, 0, Float(2));
    neo::multiply_add(left_row0, rhs, 0, acc, acc);
    REQUIRE(accumulator[0] == Catch::Approx(Float(2)));
    REQUIRE(neo::allmatch(stdex::submdspan(acc, std::tuple{1, acc.extent(0)}), is_zero));

    rhs.insert(0, 1, Float(4));
    neo::fill(acc, Float(0));
    neo::multiply_add(left_row0, rhs, 0, acc, acc);
    REQUIRE(accumulator[0] == Catch::Approx(Float(2)));
    REQUIRE(accumulator[1] == Catch::Approx(Float(4)));
    REQUIRE(neo::allmatch(stdex::submdspan(acc, std::tuple{2, acc.extent(0)}), is_zero));
}

TEMPLATE_TEST_CASE("neo/algorithm: multiply_add(csr_matrix)", "", float, double) { test_csr_matrix<TestType>(); }

#if defined(NEO_HAS_BUILTIN_FLOAT16)
TEMPLATE_TEST_CASE("neo/algorithm: multiply_add(csr_matrix)", "", _Float16) { test_csr_matrix<TestType>(); }
#endif

TEMPLATE_TEST_CASE("neo/algorithm: multiply_add(split_complex)", "", float, double)
{
    using Float = TestType;

    auto const size = GENERATE(as<std::size_t>{}, 2, 33, 128);

    auto x_buffer   = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
    auto y_buffer   = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
    auto z_buffer   = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
    auto out_buffer = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};

    auto x = neo::split_complex{
        stdex::submdspan(x_buffer.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(x_buffer.to_mdspan(), 1, stdex::full_extent),
    };
    auto y = neo::split_complex{
        stdex::submdspan(y_buffer.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(y_buffer.to_mdspan(), 1, stdex::full_extent),
    };
    auto z = neo::split_complex{
        stdex::submdspan(z_buffer.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(z_buffer.to_mdspan(), 1, stdex::full_extent),
    };
    auto out = neo::split_complex{
        stdex::submdspan(out_buffer.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(out_buffer.to_mdspan(), 1, stdex::full_extent),
    };

    neo::fill(x.real, Float(1));
    neo::fill(x.imag, Float(2));

    neo::fill(y.real, Float(3));
    neo::fill(y.imag, Float(4));

    neo::fill(z.real, Float(5));
    neo::fill(z.imag, Float(6));

    neo::multiply_add(x, y, z, out);

    for (auto i{0}; i < static_cast<int>(out.real.extent(0)); ++i) {
        REQUIRE(out.real[i] == Catch::Approx(0.0));
        REQUIRE(out.imag[i] == Catch::Approx(16.0));
    }
}
