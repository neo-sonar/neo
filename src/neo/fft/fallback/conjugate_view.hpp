// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/math/conj.hpp>

#include <concepts>

namespace neo::fft {

template<in_vector InVec>
    requires complex<typename InVec::value_type>
struct conjugate_view
{
    using value_type = typename InVec::value_type;

    constexpr explicit conjugate_view(InVec twiddles) noexcept : _twiddles{twiddles} {}

    [[nodiscard]] constexpr auto operator[](std::integral auto index) const noexcept -> value_type
    {
        return neo::math::conj(_twiddles[index]);
    }

private:
    InVec _twiddles;
};

}  // namespace neo::fft
