// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/complex.hpp>
#include <neo/fixed_point/fixed_point.hpp>
#include <neo/type_traits/always_false.hpp>

#include <array>
#include <cstdint>

namespace neo {

template<typename FixedPoint>
struct fixed_point_complex
{
    using value_type = FixedPoint;

    constexpr fixed_point_complex() = default;

    constexpr fixed_point_complex(FixedPoint re, FixedPoint im = FixedPoint{}) noexcept : _data{re, im} {}

    [[nodiscard]] constexpr auto real() const noexcept -> FixedPoint { return _data[0]; }

    [[nodiscard]] constexpr auto imag() const noexcept -> FixedPoint { return _data[1]; }

    constexpr auto real(FixedPoint re) noexcept -> void { _data[0] = re; }

    constexpr auto imag(FixedPoint im) noexcept -> void { _data[1] = im; }

    friend constexpr auto operator+(fixed_point_complex lhs, fixed_point_complex rhs) -> fixed_point_complex
    {
        return fixed_point_complex{
            lhs.real() + rhs.real(),
            lhs.imag() + rhs.imag(),
        };
    }

    friend constexpr auto operator-(fixed_point_complex lhs, fixed_point_complex rhs) -> fixed_point_complex
    {
        return fixed_point_complex{
            lhs.real() - rhs.real(),
            lhs.imag() - rhs.imag(),
        };
    }

    friend constexpr auto operator*(fixed_point_complex lhs, fixed_point_complex rhs) -> fixed_point_complex
    {
        using Int = typename FixedPoint::storage_type;

        auto const shift = [] {
            if constexpr (sizeof(Int) == 2) {
                return 17;
            } else if constexpr (sizeof(Int) == 4) {
                return 33;
            } else {
                static_assert(always_false<Int>);
            }
        }();

        auto const lre = lhs.real().value();
        auto const lim = lhs.imag().value();
        auto const rre = rhs.real().value();
        auto const rim = lhs.imag().value();

        return fixed_point_complex{
            FixedPoint{underlying_value, static_cast<Int>(((lre * rre) >> shift) - ((lim * rim) >> shift))},
            FixedPoint{underlying_value, static_cast<Int>(((lre * rim) >> shift) + ((lim * rre) >> shift))},
        };
    }

private:
    std::array<FixedPoint, 2> _data{};
};

using complex_q7  = fixed_point_complex<q7>;
using complex_q15 = fixed_point_complex<q15>;

template<typename FixedPoint>
inline constexpr auto const is_complex<fixed_point_complex<FixedPoint>> = true;

}  // namespace neo
