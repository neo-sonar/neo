// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/fixed_point/fixed_point.hpp>
#include <neo/fixed_point/simd.hpp>

#include <cassert>
#include <functional>
#include <iterator>
#include <span>

namespace neo {

namespace detail {

template<std::signed_integral Int, int FractionalBits, std::size_t Extent>
auto apply_fixed_point_kernel(
    std::span<fixed_point<Int, FractionalBits> const, Extent> lhs,
    std::span<fixed_point<Int, FractionalBits> const, Extent> rhs,
    std::span<fixed_point<Int, FractionalBits>, Extent> out,
    auto scalar_kernel,
    auto vector_kernel_s8,
    auto vector_kernel_s16
)
{
    assert(lhs.size() == rhs.size());
    assert(lhs.size() == out.size());

#if defined(NEO_HAS_ISA_SSE2) or defined(NEO_HAS_ISA_NEON)
    if constexpr (std::same_as<Int, std::int8_t>) {
        simd::apply_kernel<Int>(lhs, rhs, out, scalar_kernel, vector_kernel_s8);
        return;
    } else if constexpr (std::same_as<Int, std::int16_t>) {
        simd::apply_kernel<Int>(lhs, rhs, out, scalar_kernel, vector_kernel_s16);
        return;
    }
#endif

    for (auto i{0zu}; i < lhs.size(); ++i) {
        out[i] = scalar_kernel(lhs[i], rhs[i]);
    }
}

}  // namespace detail

/// out[i] = saturate16(lhs[i] + rhs[i])
template<std::signed_integral Int, int FractionalBits, std::size_t Extent>
auto add(
    std::span<fixed_point<Int, FractionalBits> const, Extent> lhs,
    std::span<fixed_point<Int, FractionalBits> const, Extent> rhs,
    std::span<fixed_point<Int, FractionalBits>, Extent> out
)
{
    detail::apply_fixed_point_kernel(lhs, rhs, out, std::plus{}, detail::add_kernel_s8, detail::add_kernel_s16);
}

/// out[i] = saturate16(lhs[i] - rhs[i])
template<std::signed_integral Int, int FractionalBits, std::size_t Extent>
auto subtract(
    std::span<fixed_point<Int, FractionalBits> const, Extent> lhs,
    std::span<fixed_point<Int, FractionalBits> const, Extent> rhs,
    std::span<fixed_point<Int, FractionalBits>, Extent> out
)
{
    detail::apply_fixed_point_kernel(lhs, rhs, out, std::minus{}, detail::sub_kernel_s8, detail::sub_kernel_s16);
}

/// out[i] = (lhs[i] * rhs[i]) >> FractionalBits;
template<std::signed_integral Int, int FractionalBits, std::size_t Extent>
auto multiply(
    std::span<fixed_point<Int, FractionalBits> const, Extent> lhs,
    std::span<fixed_point<Int, FractionalBits> const, Extent> rhs,
    std::span<fixed_point<Int, FractionalBits>, Extent> out
)
{
    assert(lhs.size() == rhs.size());
    assert(lhs.size() == out.size());

    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (std::same_as<Int, std::int8_t>) {
#if defined(NEO_HAS_ISA_SSE41)
        simd::apply_kernel<Int>(lhs, rhs, out, std::multiplies{}, detail::mul_kernel_s8<FractionalBits>);
        return;
#endif
    } else if constexpr (std::same_as<Int, std::int16_t>) {
#if defined(NEO_HAS_ISA_SSE41)
        simd::apply_kernel<Int>(lhs, rhs, out, std::multiplies{}, detail::mul_kernel_s16<FractionalBits>);
        return;
#elif defined(NEO_HAS_ISA_NEON)
        if constexpr (std::same_as<Int, std::int16_t> && FractionalBits == 15) {
            simd::apply_kernel<Int>(lhs, rhs, out, std::multiplies{}, detail::mul_kernel_s16);
            return;
        }
#endif
    }
    // NOLINTEND(bugprone-branch-clone)

    for (auto i{0zu}; i < lhs.size(); ++i) {
        out[i] = std::multiplies{}(lhs[i], rhs[i]);
    }
}

}  // namespace neo
