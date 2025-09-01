// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/multiply.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fft.hpp>
#include <neo/math/conj.hpp>
#include <neo/math/polar.hpp>

#include <complex>
#include <concepts>
#include <cstddef>
#include <numbers>

namespace neo::fft {

/// Bluestein FFT
/// \ingroup neo-fft
template<complex Complex>
struct fallback_dft_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit fallback_dft_plan(size_type size) : _size{size}
    {
        auto const wf = _wf.to_mdspan();
        auto const wb = _wb.to_mdspan();

        auto const n             = static_cast<Float>(size);
        auto const coef_forward  = static_cast<Float>(std::numbers::pi) / n * Float(-1);
        auto const coef_backward = static_cast<Float>(std::numbers::pi) / n * Float(1);

        for (auto i{0zu}; i < size; ++i) {
            auto const j = static_cast<Float>((i * i) % (size * 2));

            wf[i] = math::polar(Float(1), j * coef_forward);
            wb[i] = math::polar(Float(1), j * coef_backward);
        }
    }

    [[nodiscard]] auto size() const noexcept -> size_type { return _size; }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) -> void
    {
        assert(std::cmp_equal(x.extent(0), size()));

        auto const w = dir == direction::forward ? _wf.to_mdspan() : _wb.to_mdspan();
        auto const a = _a.to_mdspan();
        auto const b = _b.to_mdspan();

        // pre-processing
        fill(a, Float(0));
        fill(b, Float(0));
        multiply(x, w, stdex::submdspan(a, std::tuple{0, size()}));
        b[0] = w[0];
        for (std::size_t i{1}; i < size(); ++i) {
            auto const m = b.extent(0);
            auto const c = math::conj(w[i]);

            b[i]     = c;
            b[m - i] = c;
        }

        // convolution
        neo::fft::fft(_plan, a);
        neo::fft::fft(_plan, b);
        multiply(a, b, a);

        neo::fft::ifft(_plan, a);
        scale(Float(1) / Float(_plan.size()), a);

        // post-processing
        multiply(stdex::submdspan(a, std::tuple{0, size()}), w, x);
    }

private:
    using Float = typename Complex::value_type;

    size_type _size;
    fft_plan<Complex> _plan{
        fft::from_order,
        fft::next_order(size() * size_type(2) + size_type(1)),
    };

    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _wf{size()};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _wb{size()};

    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _a{_plan.size()};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _b{_plan.size()};
};

}  // namespace neo::fft
