// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <complex>
#include <concepts>

namespace neo_math_adl_check_real {

using std::real;

auto real(auto const&) -> void = delete;

template<typename T>
concept has_member_real = requires(T const& t) { t.real(); };

template<typename T>
concept has_adl_real = not has_member_real<T> and requires(T const& t) { real(t); };

struct fn
{
    template<std::floating_point T>
    [[nodiscard]] constexpr auto operator()(T x) const noexcept
    {
        return x;
    }

    template<has_member_real T>
    [[nodiscard]] constexpr auto operator()(T x) const noexcept
    {
        return x.real();
    }

    template<has_adl_real T>
        requires(not std::floating_point<T>)
    [[nodiscard]] constexpr auto operator()(T x) const noexcept
    {
        return real(x);
    }
};

}  // namespace neo_math_adl_check_real

namespace neo::math {

/// \ingroup neo-math
inline constexpr auto const real = neo_math_adl_check_real::fn{};

}  // namespace neo::math
