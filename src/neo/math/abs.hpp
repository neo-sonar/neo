// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <concepts>

namespace neo_math_adl_check_abs {

using std::abs;

auto abs(auto const&) -> void = delete;

template<typename T>
concept has_member_abs = requires(T const& t) { t.abs(); };

template<typename T>
concept has_adl_abs = not has_member_abs<T> and requires(T const& t) { abs(t); };

struct fn
{
    template<std::unsigned_integral UInt>
    [[nodiscard]] constexpr auto operator()(UInt x) const noexcept
    {
        return x;
    }

    template<std::signed_integral UInt>
    [[nodiscard]] constexpr auto operator()(UInt x) const noexcept
    {
        return std::abs(x);
    }

    template<has_member_abs T>
    [[nodiscard]] constexpr auto operator()(T x) const noexcept
    {
        return x.abs();
    }

    template<has_adl_abs T>
        requires(not std::signed_integral<T>)
    [[nodiscard]] constexpr auto operator()(T x) const noexcept
    {
        return abs(x);
    }
};

}  // namespace neo_math_adl_check_abs

namespace neo::math {

/// \ingroup neo-math
inline constexpr auto const abs = neo_math_adl_check_abs::fn{};

}  // namespace neo::math
