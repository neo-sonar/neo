// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/backend/bluestein.hpp>
#include <neo/fft/direction.hpp>

#if defined(NEO_HAS_INTEL_IPP)
    #include <neo/fft/backend/ipp.hpp>
#endif

#include <cassert>
#include <numbers>

namespace neo::fft {

#if defined(NEO_HAS_INTEL_IPP)
template<complex Complex>
using dft_plan = intel_ipp_dft_plan<Complex>;
#else
template<complex Complex>
using dft_plan = bluestein_plan<Complex>;
#endif

template<in_vector InVec, out_vector OutVec>
    requires std::same_as<typename InVec::value_type, typename OutVec::value_type>
auto dft(InVec in, OutVec out, direction dir = direction::forward) -> void
{
    using Complex = typename OutVec::value_type;
    using Float   = typename Complex::value_type;

    assert(neo::detail::extents_equal(in, out));

    static constexpr auto const two_pi = static_cast<Float>(std::numbers::pi * 2.0);

    auto const sign = dir == direction::forward ? Float(-1) : Float(1);
    auto const size = in.extent(0);

    for (std::size_t k = 0; k < size; ++k) {
        auto tmp = Complex{};
        for (std::size_t n = 0; n < size; ++n) {
            auto const input = in(n);
            auto const w     = std::polar(Float(1), sign * two_pi * Float(n) * Float(k) / Float(size));
            tmp += input * w;
        }
        out(k) = tmp;
    }
}

template<typename Plan, inout_vector Vec>
constexpr auto dft(Plan& plan, Vec inout) -> void
{
    plan(inout, direction::forward);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
constexpr auto dft(Plan& plan, InVec input, OutVec output) -> void
{
    if constexpr (requires { plan(input, output, direction::forward); }) {
        plan(input, output, direction::forward);
    } else {
        copy(input, output);
        dft(plan, output);
    }
}

template<typename Plan, inout_vector Vec>
constexpr auto idft(Plan& plan, Vec inout) -> void
{
    plan(inout, direction::backward);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
constexpr auto idft(Plan& plan, InVec input, OutVec output) -> void
{
    if constexpr (requires { plan(input, output, direction::backward); }) {
        plan(input, output, direction::backward);
    } else {
        copy(input, output);
        idft(plan, output);
    }
}

}  // namespace neo::fft
