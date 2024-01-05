// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <complex>
#include <concepts>

namespace neo::math {

namespace detail {

using std::conj;

auto conj(auto const&) -> void = delete;

template<typename T>
concept has_member_conj = requires(T const& t) { t.conj(); };

template<typename T>
concept has_adl_conj = not has_member_conj<T> and requires(T const& t) { conj(t); };

struct conj_fn
{
    template<typename T>
        requires has_member_conj<T>
    auto operator()(T x) const noexcept
    {
        return x.conj();
    }

    template<typename T>
        requires has_adl_conj<T>
    auto operator()(T x) const noexcept
    {
        return conj(x);
    }
};

}  // namespace detail

inline constexpr auto const conj = detail::conj_fn{};

}  // namespace neo::math
