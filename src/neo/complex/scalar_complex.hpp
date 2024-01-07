// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/complex/complex.hpp>

#include <cmath>
#include <tuple>

namespace neo {

template<typename Scalar>
struct scalar_complex
{
    using value_type = Scalar;

    constexpr scalar_complex() = default;

    constexpr scalar_complex(Scalar re, Scalar im = Scalar{}) noexcept : _data{re, im} {}

    [[nodiscard]] constexpr auto real() const noexcept -> Scalar { return _data[0]; }

    [[nodiscard]] constexpr auto imag() const noexcept -> Scalar { return _data[1]; }

    constexpr auto real(Scalar re) noexcept -> void { _data[0] = re; }

    constexpr auto imag(Scalar im) noexcept -> void { _data[1] = im; }

    friend constexpr auto operator+(scalar_complex lhs, scalar_complex rhs) noexcept -> scalar_complex
    {
        return scalar_complex{
            lhs.real() + rhs.real(),
            lhs.imag() + rhs.imag(),
        };
    }

    friend constexpr auto operator-(scalar_complex lhs, scalar_complex rhs) noexcept -> scalar_complex
    {
        return scalar_complex{
            lhs.real() - rhs.real(),
            lhs.imag() - rhs.imag(),
        };
    }

    friend constexpr auto operator*(scalar_complex lhs, scalar_complex rhs) noexcept -> scalar_complex
    {
        return scalar_complex{
            lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
            lhs.real() * rhs.imag() + lhs.imag() * rhs.real(),
        };
    }

    friend constexpr auto operator/(scalar_complex lhs, scalar_complex rhs) noexcept -> scalar_complex
    {
        auto const a     = lhs.real();
        auto const b     = lhs.imag();
        auto const c     = rhs.real();
        auto const d     = rhs.imag();
        auto const denom = c * c + d * d;
        return scalar_complex{
            (a * c + b * d) / denom,
            (b * c - a * d) / denom,
        };
    }

    template<typename OtherScalar>
        requires(not complex<OtherScalar>)
    friend constexpr auto operator*=(scalar_complex& lhs, OtherScalar rhs) noexcept -> scalar_complex&
    {
        lhs = scalar_complex{lhs.real() * rhs, lhs.imag() * rhs};
        return lhs;
    }

    template<typename OtherScalar>
        requires(not complex<OtherScalar>)
    friend constexpr auto operator/=(scalar_complex& lhs, OtherScalar rhs) noexcept -> scalar_complex&
    {
        lhs = scalar_complex{lhs.real() / rhs, lhs.imag() / rhs};
        return lhs;
    }

private:
    Scalar _data[2];  // NOLINT
};

template<typename Scalar>
inline constexpr auto const is_complex<scalar_complex<Scalar>> = true;

template<std::size_t I, typename T>
    requires(I < 2)
[[nodiscard]] constexpr auto get(scalar_complex<T> const& z) noexcept -> T
{
    if constexpr (I == 0) {
        return z.real();
    } else {
        return z.imag();
    }
}

template<typename Scalar>
[[nodiscard]] constexpr auto conj(scalar_complex<Scalar> const& z) noexcept -> scalar_complex<Scalar>
{
    return {z.real(), -z.imag()};
}

template<typename Scalar>
[[nodiscard]] constexpr auto abs(scalar_complex<Scalar> const& z) noexcept -> Scalar
{
    auto const dot = [](auto re, auto im) {
        auto const sum = re * re + im * im;
#if defined(NEO_HAS_BUILTIN_FLOAT16)
        if constexpr (std::same_as<Scalar, _Float16>) {
            return static_cast<float>(sum);
        } else {
            return sum;
        }
#else
        return sum;
#endif
    };

    return static_cast<Scalar>(std::sqrt(dot(z.real(), z.imag())));
}

#if defined(NEO_HAS_BUILTIN_FLOAT16)
using complex32 = scalar_complex<_Float16>;
#endif
using complex64  = scalar_complex<float>;
using complex128 = scalar_complex<double>;

}  // namespace neo

template<typename T>
struct std::tuple_size<neo::scalar_complex<T>> : std::integral_constant<std::size_t, 2>
{};

template<std::size_t I, typename T>
    requires(I < 2)
struct std::tuple_element<I, neo::scalar_complex<T>>
{
    using type = T;
};
