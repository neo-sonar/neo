// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <complex>
#include <concepts>

namespace neo_math_adl_check_conj {

using std::conj;

auto conj(auto const&) -> void = delete;

template<typename T>
concept has_member_conj = requires(T const& t) { t.conj(); };

template<typename T>
concept has_adl_conj = not has_member_conj<T> and requires(T const& t) { conj(t); };

struct fn
{
    template<has_member_conj T>
    auto operator()(T x) const noexcept
    {
        return x.conj();
    }

    template<has_adl_conj T>
    auto operator()(T x) const noexcept
    {
        return conj(x);
    }
};

}  // namespace neo_math_adl_check_conj

namespace neo::math {

/// \ingroup neo-math
inline constexpr auto const conj = neo_math_adl_check_conj::fn{};

}  // namespace neo::math
