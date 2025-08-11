// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/complex/complex.hpp>
#include <neo/fft/direction.hpp>
#include <neo/math/polar.hpp>

#include <concepts>
#include <numbers>

namespace neo::fft {

/// \ingroup neo-fft
template<complex Complex>
[[nodiscard]] constexpr auto twiddle(std::integral auto size, std::integral auto index, direction dir) noexcept
    -> Complex
{
    using Float = typename Complex::value_type;

    auto const sign   = dir == direction::forward ? Float(-1) : Float(1);
    auto const two_pi = static_cast<Float>(std::numbers::pi * 2.0);
    auto const angle  = sign * two_pi * Float(index) / Float(size);

    auto const w = math::polar(Float(1), angle);  // returns std::complex
    return Complex{w.real(), w.imag()};           // convert to custom complex (maybe)
}

/// \ingroup neo-fft
template<inout_vector OutVec>
auto fill_twiddle_lut_radix2(OutVec lut, direction dir) noexcept -> void
{
    using Complex = typename OutVec::value_type;

    auto const lut_size = lut.size();
    auto const size     = lut_size * 2ULL;

    for (std::size_t i = 0; i < lut_size; ++i) {
        lut[i] = twiddle<Complex>(size, i, dir);
    }
}

/// \ingroup neo-fft
template<complex Complex>
auto make_twiddle_lut_radix2(std::size_t size, direction dir)
{
    auto lut = stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>>{size / 2U};
    fill_twiddle_lut_radix2(lut.to_mdspan(), dir);
    return lut;
}

}  // namespace neo::fft
