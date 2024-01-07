// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <complex>
#include <concepts>

namespace neo::math {

namespace detail {

using std::imag;

auto imag(auto const&) -> void = delete;

template<typename T>
concept has_member_imag = requires(T const& t) { t.imag(); };

template<typename T>
concept has_adl_imag = not has_member_imag<T> and requires(T const& t) { imag(t); };

struct imag_fn
{
    template<std::floating_point T>
    [[nodiscard]] constexpr auto operator()(T x) const noexcept
    {
        return x;
    }

    template<has_member_imag T>
    [[nodiscard]] constexpr auto operator()(T x) const noexcept
    {
        return x.imag();
    }

    template<has_adl_imag T>
        requires(not std::floating_point<T>)
    [[nodiscard]] constexpr auto operator()(T x) const noexcept
    {
        return imag(x);
    }
};

}  // namespace detail

inline constexpr auto const imag = detail::imag_fn{};

}  // namespace neo::math
