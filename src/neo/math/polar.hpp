// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <complex>
#include <concepts>

namespace neo_math_adl_check_polar {

using std::polar;

auto polar(auto const&, auto const&) -> void = delete;

template<typename T>
concept has_adl_polar = requires(T const& r, T const& theta) { polar(r, theta); };

struct fn
{
    template<has_adl_polar T>
    [[nodiscard]] constexpr auto operator()(T r, T theta) const
    {
        return polar(r, theta);
    }
};

}  // namespace neo_math_adl_check_polar

namespace neo::math {

/// \ingroup neo-math
inline constexpr auto const polar = neo_math_adl_check_polar::fn{};

}  // namespace neo::math
