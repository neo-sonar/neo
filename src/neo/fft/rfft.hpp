// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/fft/fallback/fallback_rfft_plan.hpp>
#include <neo/fft/fft.hpp>
#include <neo/math/conj.hpp>
#include <neo/math/imag.hpp>
#include <neo/math/real.hpp>
#include <neo/type_traits/value_type_t.hpp>

namespace neo::fft {

#if defined(NEO_HAS_INTEL_IPP)
/// \ingroup neo-fft
template<std::floating_point Float, typename Complex = std::complex<Float>>
using rfft_plan = intel_ipp_rfft_plan<Float, Complex>;
#else
/// \ingroup neo-fft
template<std::floating_point Float, typename Complex = std::complex<Float>>
using rfft_plan = fallback_rfft_plan_v2<Float, Complex>;
#endif

/// \ingroup neo-fft
template<typename Plan, in_vector InVec, out_vector OutVec>
    requires(std::floating_point<value_type_t<InVec>> and complex<value_type_t<OutVec>>)
constexpr auto rfft(Plan& plan, InVec input, OutVec output)
{
    return plan(input, output);
}

/// \ingroup neo-fft
template<typename Plan, in_vector InVec, out_vector OutVec>
    requires(complex<value_type_t<InVec>> and std::floating_point<value_type_t<OutVec>>)
constexpr auto irfft(Plan& plan, InVec input, OutVec output)
{
    return plan(input, output);
}

/// \ingroup neo-fft
template<in_vector InVec, out_vector OutVecX, out_vector OutVecY>
    requires(complex<value_type_t<InVec>> and complex<value_type_t<OutVecX>> and complex<value_type_t<OutVecY>>)
auto rfft_deinterleave(InVec dft, OutVecX x, OutVecY y) -> void
{
    using Complex = value_type_t<InVec>;
    using Float   = value_type_t<Complex>;

    auto const n = static_cast<int>(dft.extent(0));
    auto const i = Complex{Float(0), Float(-1)};

    x[0] = math::real(dft[0]);
    y[0] = math::imag(dft[0]);

    for (auto k{1}; k < n / 2 + 1; ++k) {
        auto const zk  = dft[k];
        auto const znk = math::conj(dft[n - k]);

        x[k] = (zk + znk) * Float(0.5);
        y[k] = ((zk - znk) * i) * Float(0.5);
    }
}

}  // namespace neo::fft
