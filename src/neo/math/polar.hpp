// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <complex>
#include <concepts>

namespace neo::math {

namespace detail {

using std::polar;

auto polar(auto const&, auto const&) -> void = delete;

template<typename T>
concept has_adl_polar = requires(T const& r, T const& theta) { polar(r, theta); };

struct polar_fn
{
    template<has_adl_polar T>
    auto operator()(T r, T theta) const noexcept
    {
        return polar(r, theta);
    }
};

}  // namespace detail

inline constexpr auto const polar = detail::polar_fn{};

}  // namespace neo::math
