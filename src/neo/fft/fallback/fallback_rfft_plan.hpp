// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fft.hpp>
#include <neo/fft/order.hpp>
#include <neo/math/conj.hpp>

namespace neo::fft {

/// \ingroup neo-fft
template<typename Float, typename Complex = std::complex<Float>>
struct fallback_rfft_plan_v1
{
    using real_type    = Float;
    using complex_type = Complex;
    using size_type    = std::size_t;

    fallback_rfft_plan_v1(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<in_vector_of<Float> InVec, out_vector_of<Complex> OutVec>
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        auto const buf    = _buffer.to_mdspan();
        auto const coeffs = size() / 2 + 1;

        copy(in, buf);
        _fft(buf, direction::forward);
        copy(stdex::submdspan(buf, std::tuple{0zu, coeffs}), stdex::submdspan(out, std::tuple{0zu, coeffs}));
    }

    template<in_vector_of<Complex> InVec, out_vector_of<Float> OutVec>
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        auto const buf    = _buffer.to_mdspan();
        auto const coeffs = size() / 2 + 1;

        copy(in, stdex::submdspan(buf, std::tuple{0, in.extent(0)}));

        // Fill upper half with conjugate
        for (auto i{coeffs}; i < size(); ++i) {
            buf[i] = math::conj(buf[size() - i]);
        }

        _fft(buf, direction::backward);
        for (auto i{0zu}; i < size(); ++i) {
            out[i] = buf[i].real();
        }
    }

private:
    size_type _order;
    fft_plan<Complex> _fft{from_order, _order};
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _buffer{size()};
};

/// \ingroup neo-fft
template<typename Float, typename Complex = std::complex<Float>>
struct fallback_rfft_plan_v2
{
    using real_type    = Float;
    using complex_type = Complex;
    using size_type    = std::size_t;

    fallback_rfft_plan_v2(from_order_tag /*tag*/, size_type order) : _order{validate_order(order)} {}

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<in_vector_of<Float> InVec, out_vector_of<Complex> OutVec>
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        auto const m   = _fft.size();
        auto const w   = _w.to_mdspan();
        auto const tmp = _tmp.to_mdspan();

        // 1) Pack: z[m] = in[2m] + j*in[2m+1]
        for (auto i = 0zu; i < _fft.size(); ++i) {
            auto idx = 2zu * i;
            tmp(i)   = Complex{in[idx], in[idx + 1zu]};
        }

        // 2) FFT size M: Z = FFT(z)
        _fft(tmp, direction::forward);

        // 5) Endpoints from Z[0] = a0 + j*b0
        auto const a0 = tmp[0].real();
        auto const b0 = tmp[0].imag();

        out[0] = Complex{a0 + b0};
        out[m] = Complex{a0 - b0};

        // 3&4) For k=1..M-1
        for (auto k = 1zu; k < m; ++k) {
            auto const a  = tmp[k];
            auto const bc = math::conj(tmp[m - k]);

            // E = 0.5*(A + b)
            auto e = Float(0.5) * (a + bc);

            // O = -0.5j*(A - b)  == (0.5*Δi) + j*(-0.5*Δr)
            auto delta = a - bc;
            auto o     = Complex(Float(0.5) * delta.imag(), Float(-0.5) * delta.real());

            // X[k] = E + W_k * O  (unique half only)
            out[k] = e + w[k] * o;
        }
    }

    template<in_vector_of<Complex> InVec, out_vector_of<Float> OutVec>
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        auto const m   = _fft.size();
        auto const w   = _w.to_mdspan();
        auto const tmp = _tmp.to_mdspan();

        // 1) Z[0] from DC & Nyquist
        auto const s0  = in[0].real();
        auto const s_m = in[m].real();
        tmp[0]         = Complex{s0 + s_m, s0 - s_m} * Float(0.5);

        // 2) Rebuild Z[k], Z[M-k] for k=1..M-1
        for (auto k = 1zu; k < m; ++k) {
            auto const a  = in[k];
            auto const bc = math::conj(in[m - k]);

            // s = 0.5*(A + B^*)
            // T = 0.5*(A - B^*) / W_k  = 0.5*(A - B^*) * conj(W_k)
            auto const s = Float(0.5) * (a + bc);
            auto const t = Float(0.5) * (a - bc) * math::conj(w[k]);

            // Z[k] = s + j*t   => Re = s.r - t.i,  Im = s.i + t.r
            // Z[m-k] = conj(s - j*t)
            tmp[k]     = Complex{s.real() - t.imag(), s.imag() + t.real()};
            tmp[m - k] = math::conj(Complex{s.real() + t.imag(), s.imag() - t.real()});
        }

        // 3) IFFT size M: z = IFFT(Z)
        _fft(tmp, direction::backward);

        // 4) Unpack and scale
        for (auto i = 0zu; i < m; ++i) {
            auto const idx = 2zu * i;
            auto const y   = tmp[i] * Float(2);
            out[idx]       = y.real();
            out[idx + 1zu] = y.imag();
        }
    }

private:
    static auto validate_order(size_type order) -> size_type
    {
        if (order == 0) {
            throw std::runtime_error{"invalid order for rfft_plan. order = " + std::to_string(order)};
        }
        return order;
    }

    // Precompute twiddles W_k = exp(-j*2*pi*k/N)
    static auto make_twiddles(size_type n) -> stdex::mdarray<Complex, stdex::dextents<size_type, 1>>
    {
        auto const two_pi_over_n = static_cast<Float>(2.0 * std::numbers::pi) / static_cast<Float>(n);

        auto w = stdex::mdarray<Complex, stdex::dextents<size_type, 1>>{n / 2 + 1};
        for (auto k = 0zu; k <= n / 2zu; ++k) {
            auto const ang = -two_pi_over_n * static_cast<Float>(k);
            auto const z   = std::polar(Float(1), ang);
            w(k)           = Complex{z.real(), z.imag()};
        }

        return w;
    }

    size_type _order;
    fft_plan<Complex> _fft{from_order, _order - 1zu};
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _w{make_twiddles(size())};
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _tmp{_fft.size()};
};

}  // namespace neo::fft
