// SPDX-License-Identifier: MIT
#include "simd.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <array>

template<typename Batch>
static auto test()
{
    using Float = typename Batch::value_type;

    auto test = [](auto op, auto left_scalar, auto right_scalar, auto expected) {
        auto left  = std::array<Float, Batch::size>{};
        auto right = std::array<Float, Batch::size>{};

        std::fill(left.begin(), left.end(), left_scalar);
        std::fill(right.begin(), right.end(), right_scalar);

        auto output = std::array<Float, Batch::size>{};
        auto result = op(Batch::load_unaligned(left.data()), Batch::load_unaligned(right.data()));
        result.store_unaligned(output.data());

        for (auto val : output) {
            REQUIRE(val == Catch::Approx(expected));
        }
    };

    test(std::plus{}, Float(1), Float(2), Float(3));
    test(std::minus{}, Float(1), Float(2), Float(-1));
    test(std::multiplies{}, Float(1), Float(2), Float(2));
}

#if defined(NEO_HAS_BUILTIN_FLOAT16) and defined(NEO_HAS_SIMD_F16C)
TEMPLATE_TEST_CASE("neo/simd: batch", "", neo::float16x8, neo::float16x16) { test<TestType>(); }
#endif

#if defined(NEO_HAS_SIMD_SSE2)
TEMPLATE_TEST_CASE("neo/simd: batch", "", neo::float32x4, neo::float64x2) { test<TestType>(); }
#endif

#if defined(NEO_HAS_SIMD_AVX)
TEMPLATE_TEST_CASE("neo/simd: batch", "", neo::float32x8, neo::float64x4) { test<TestType>(); }
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
TEMPLATE_TEST_CASE("neo/simd: batch", "", neo::float32x16, neo::float64x8) { test<TestType>(); }
#endif
