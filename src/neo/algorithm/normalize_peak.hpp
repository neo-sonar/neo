// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/scale.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/math/abs.hpp>
#include <neo/math/float_equality.hpp>
#include <neo/type_traits/value_type_t.hpp>

#include <algorithm>
#include <functional>
#include <limits>

namespace neo {

template<in_object InObj>
    requires std::floating_point<value_type_t<InObj>>
[[nodiscard]] auto normalize_peak_factor(InObj obj) noexcept
{
    using Float = value_type_t<InObj>;

    auto abs_max = Float(0);
    if constexpr (InObj::rank() == 1) {
        if (obj.extent(0) == 0) {
            return Float(1);
        }

        for (decltype(obj.extent(0)) i{0}; i < obj.extent(0); ++i) {
            abs_max = std::max(abs_max, math::abs(obj[i]));
        }

    } else {
        if (obj.extent(0) == 0 and obj.extent(1) == 0) {
            return Float(1);
        }

        for (decltype(obj.extent(0)) i{0}; i < obj.extent(0); ++i) {
            for (decltype(obj.extent(1)) j{0}; j < obj.extent(1); ++j) {
                abs_max = std::max(abs_max, math::abs(obj(i, j)));
            }
        }
    }

    if (float_equality::exact(abs_max, Float(0))) {
        return Float(1);
    }

    return Float(1) / math::abs(abs_max);
}

// normalized_sample = sample / max(abs(buffer))
template<inout_object InOutObj>
auto normalize_peak(InOutObj obj) noexcept -> void
{
    scale(normalize_peak_factor(obj), obj);
}

}  // namespace neo
