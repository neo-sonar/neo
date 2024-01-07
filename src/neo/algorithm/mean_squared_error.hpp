// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/math/abs.hpp>

#include <cassert>
#include <concepts>

namespace neo {

template<in_object InObjL, in_object InObjR>
    requires(InObjL::rank() == InObjR::rank())
[[nodiscard]] auto mean_squared_error(InObjL lhs, InObjR rhs) noexcept
{
    using Index  = std::common_type_t<typename InObjL::index_type, typename InObjR::index_type>;
    using Scalar = std::common_type_t<typename InObjL::value_type, typename InObjR::value_type>;

    auto abs_if_needed = [](auto val) {
        if constexpr (complex<Scalar>) {
            return math::abs(val);
        } else {
            return val;
        }
    };

    using Float = decltype(abs_if_needed(std::declval<Scalar>()));

    static_assert(std::floating_point<Float>);
    assert(lhs.extents() == rhs.extents());

    auto sum = Float(0);

    if constexpr (InObjL::rank() == 1) {
        for (Index i{0}; std::cmp_less(i, lhs.extent(0)); ++i) {
            auto const diff    = abs_if_needed(lhs[i]) - abs_if_needed(rhs[i]);
            auto const squared = diff * diff;
            sum += squared;
        }
    } else {
        for (Index i{0}; std::cmp_less(i, lhs.extent(0)); ++i) {
            for (Index j{0}; std::cmp_less(j, lhs.extent(1)); ++j) {
                auto const diff    = abs_if_needed(lhs(i, j)) - abs_if_needed(rhs(i, j));
                auto const squared = diff * diff;
                sum += squared;
            }
        }
    }

    return sum / static_cast<Float>(lhs.size());
}

}  // namespace neo
