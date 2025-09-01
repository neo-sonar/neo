// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/order.hpp>
#include <neo/fft/reference/digitrevorder.hpp>
#include <neo/fft/twiddle.hpp>
#include <neo/math/ipow.hpp>

namespace neo::fft {

/// \brief C2C Cooley–Tukey Radix-4
/// \ingroup neo-fft
template<complex Complex, bool UseDIT>
struct c2c_radix4_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit c2c_radix4_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{11}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return ipow<size_type(4)>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return ipow<size_type(4)>(order()); }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        using Float = value_type_t<Complex>;

        auto const sign = dir == direction::forward ? Float(-1.0) : Float(1.0);
        auto const w    = dir == direction::forward ? stdex::submdspan(_w.to_mdspan(), 0, stdex::full_extent)
                                                    : stdex::submdspan(_w.to_mdspan(), 1, stdex::full_extent);

        if constexpr (UseDIT) {
            _reorder(x);
            c2c_dit4(x, w, static_cast<std::size_t>(_order), sign);
        } else {
            c2c_dif4(x, w, static_cast<std::size_t>(_order), sign);
            _reorder(x);
        }
    }

private:
    static auto c2c_dit4(
        inout_vector_of<Complex> auto x,
        in_vector_of<Complex> auto twiddle,
        std::size_t order,
        std::floating_point auto sign
    ) -> void
    {
        using Float = value_type_t<Complex>;

        auto const z = Complex{Float(0), sign};

        auto length   = 4zu;
        auto w_stride = ipow<4zu>(order - 1zu);
        auto krange   = 1zu;
        auto block    = x.size() / 4zu;
        auto base     = 0zu;

        for (auto o{0zu}; o < order; ++o) {
            auto const offset = length / 4;

            for (auto h{0zu}; h < block; ++h) {

                {
                    auto const i0 = base;
                    auto const i1 = base + (1 * offset);
                    auto const i2 = base + (2 * offset);
                    auto const i3 = base + (3 * offset);

                    auto const c0 = x[i0];
                    auto const c1 = x[i1];
                    auto const c2 = x[i2];
                    auto const c3 = x[i3];

                    auto const d0 = c0 + c2;
                    auto const d1 = c0 - c2;
                    auto const d2 = c1 + c3;
                    auto const d3 = (c1 - c3) * z;

                    x[i0] = d0 + d2;
                    x[i1] = d1 - d3;
                    x[i2] = -d2 + d0;
                    x[i3] = d3 + d1;
                }

                for (auto k{1zu}; k < krange; ++k) {
                    auto w1 = twiddle[1 * k * w_stride];
                    auto w2 = twiddle[2 * k * w_stride];
                    auto w3 = twiddle[3 * k * w_stride];

                    auto const i0 = base + k;
                    auto const i1 = base + k + (1 * offset);
                    auto const i2 = base + k + (2 * offset);
                    auto const i3 = base + k + (3 * offset);

                    auto const c0 = x[i0];
                    auto const c1 = (x[i1] * w1);
                    auto const c2 = (x[i2] * w2);
                    auto const c3 = (x[i3] * w3);

                    auto const d0 = c0 + c2;
                    auto const d1 = c0 - c2;
                    auto const d2 = c1 + c3;
                    auto const d3 = (c1 - c3) * z;

                    x[i0] = d0 + d2;
                    x[i1] = d1 - d3;
                    x[i2] = -d2 + d0;
                    x[i3] = d3 + d1;
                }

                base = base + (4zu * krange);
            }

            block    = block / 4zu;
            length   = 4 * length;
            krange   = 4 * krange;
            base     = 0;
            w_stride = w_stride / 4;
        }
    }

    static auto c2c_dif4(
        inout_vector_of<Complex> auto x,
        in_vector_of<Complex> auto twiddle,
        std::size_t order,
        std::floating_point auto sign
    ) -> void
    {
        using Float = value_type_t<Complex>;

        auto const z = Complex{Float(0), sign};

        auto length   = ipow<4zu>(order);
        auto w_stride = 1zu;
        auto krange   = length / 4zu;
        auto block    = 1zu;
        auto base     = 0zu;

        for (auto o{0zu}; o < order; ++o) {
            for (auto h{0zu}; h < block; ++h) {
                for (auto k{0zu}; k < krange; ++k) {
                    auto const offset = length / 4zu;
                    auto const i0     = base + k;
                    auto const i1     = base + k + offset;
                    auto const i2     = base + k + (2 * offset);
                    auto const i3     = base + k + (3 * offset);

                    auto const d0 = x[i0] + x[i2];
                    auto const d1 = x[i0] - x[i2];
                    auto const d2 = x[i1] + x[i3];
                    auto const d3 = x[i1] - x[i3];
                    x[i0]         = d0 + d2;

                    if (k == 0) {
                        x[i1] = d1 - (z * d3);
                        x[i2] = d0 - d2;
                        x[i3] = d1 + (z * d3);
                    } else {
                        auto w1 = twiddle[1 * k * w_stride];
                        auto w2 = twiddle[2 * k * w_stride];
                        auto w3 = twiddle[3 * k * w_stride];
                        x[i1]   = (d1 - (z * d3)) * w1;
                        x[i2]   = (d0 - d2) * w2;
                        x[i3]   = (d1 + (z * d3)) * w3;
                    }
                }

                base = base + (4zu * krange);
            }

            block    = block * 4zu;
            length   = length / 4zu;
            krange   = krange / 4zu;
            base     = 0;
            w_stride = w_stride * 4;
        }
    }

    [[nodiscard]] static auto make_twiddle_lut(size_type n) -> stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>>
    {
        auto kmax  = 3zu * (n / 4zu - 1zu);
        auto w     = stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>>{2zu, n};
        auto w_fwd = stdex::submdspan(w.to_mdspan(), 0, stdex::full_extent);
        auto w_bwd = stdex::submdspan(w.to_mdspan(), 1, stdex::full_extent);
        for (auto i{0zu}; i < kmax + 1zu; ++i) {
            w_fwd[i] = twiddle<Complex>(n, i, direction::forward);
            w_bwd[i] = twiddle<Complex>(n, i, direction::backward);
        }
        return w;
    }

    size_type _order;
    digitrevorder_plan<4> _reorder{size()};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>> _w{make_twiddle_lut(size())};
};

/// \brief C2C Cooley–Tukey Radix-4 DIT
/// \ingroup neo-fft
template<typename Complex>
using c2c_dit4_plan = c2c_radix4_plan<Complex, true>;

/// \brief C2C Cooley–Tukey Radix-4 DIF
/// \ingroup neo-fft
template<typename Complex>
using c2c_dif4_plan = c2c_radix4_plan<Complex, false>;

}  // namespace neo::fft
