// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/allmatch.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/math/abs.hpp>
#include <neo/type_traits/value_type_t.hpp>

#include <algorithm>
#include <concepts>

namespace neo {

template<in_object InObj1, in_object InObj2, typename Scalar>
[[nodiscard]] auto allclose(InObj1 lhs, InObj2 rhs, Scalar tolerance) noexcept -> bool
{
    return allmatch(lhs, rhs, [tolerance](auto const& left, auto const& right) -> bool {
        return math::abs(left - right) <= tolerance;
    });
}

template<in_object InObj1, in_object InObj2>
[[nodiscard]] auto allclose(InObj1 lhs, InObj2 rhs) noexcept -> bool
{
    auto const tolerance = [] {
        using Left  = value_type_t<InObj1>;
        using Right = value_type_t<InObj2>;
        using Float = decltype(math::abs(Left{} - Right{}));

        static_assert(std::floating_point<Float>);

        if constexpr (std::same_as<Float, float>) {
            return Float(1e-5);
        } else {
            return Float(1e-9);
        }
    }();

    return allclose(lhs, rhs, tolerance);
}

}  // namespace neo
