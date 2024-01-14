// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/experimental/digitrevorder.hpp>
#include <neo/fft/order.hpp>
#include <neo/fft/twiddle.hpp>
#include <neo/math/ipow.hpp>

namespace neo::fft::experimental {

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

        auto length = 4UL;
        auto tss    = ipow<size_type(4)>(order - 1UL);
        auto krange = 1UL;
        auto block  = x.size() / 4UL;
        auto base   = 0UL;

        for (auto o{0ULL}; o < order; ++o) {
            for (auto h{0ULL}; h < block; ++h) {
                for (auto k{0ULL}; k < krange; ++k) {
                    auto const offset = length / 4;
                    auto const avar   = base + k;
                    auto const bvar   = base + k + offset;
                    auto const cvar   = base + k + (2 * offset);
                    auto const dvar   = base + k + (3 * offset);

                    auto xbr1 = Complex{};
                    auto xcr2 = Complex{};
                    auto xdr3 = Complex{};
                    if (k == 0) {
                        xbr1 = x[bvar];
                        xcr2 = x[cvar];
                        xdr3 = x[dvar];
                    } else {
                        auto r1var = twiddle[k * tss];
                        auto r2var = twiddle[2 * k * tss];
                        auto r3var = twiddle[3 * k * tss];
                        xbr1       = (x[bvar] * r1var);
                        xcr2       = (x[cvar] * r2var);
                        xdr3       = (x[dvar] * r3var);
                    }

                    auto const evar = x[avar] + xcr2;
                    auto const fvar = x[avar] - xcr2;
                    auto const gvar = xbr1 + xdr3;
                    auto const hh   = xbr1 - xdr3;
                    auto const j_h  = z * hh;

                    x[avar] = evar + gvar;
                    x[bvar] = fvar - j_h;
                    x[cvar] = -gvar + evar;
                    x[dvar] = j_h + fvar;
                }

                base = base + (4UL * krange);
            }

            block  = block / 4UL;
            length = 4 * length;
            krange = 4 * krange;
            base   = 0;
            tss    = tss / 4;
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

        auto length = ipow<size_type(4)>(order);
        auto tss    = 1UL;
        auto krange = length / 4UL;
        auto block  = 1UL;
        auto base   = 0UL;

        for (auto o{0ULL}; o < order; ++o) {
            for (auto h{0ULL}; h < block; ++h) {
                for (auto k{0ULL}; k < krange; ++k) {
                    auto const offset = length / 4UL;
                    auto const a      = base + k;
                    auto const b      = base + k + offset;
                    auto const c      = base + k + (2 * offset);
                    auto const d      = base + k + (3 * offset);
                    auto const apc    = x[a] + x[c];
                    auto const bpd    = x[b] + x[d];
                    auto const amc    = x[a] - x[c];
                    auto const bmd    = x[b] - x[d];
                    x[a]              = apc + bpd;

                    if (k == 0) {
                        x[b] = amc - (z * bmd);
                        x[c] = apc - bpd;
                        x[d] = amc + (z * bmd);
                    } else {
                        auto r1 = twiddle[k * tss];
                        auto r2 = twiddle[2 * k * tss];
                        auto r3 = twiddle[3 * k * tss];
                        x[b]    = (amc - (z * bmd)) * r1;
                        x[c]    = (apc - bpd) * r2;
                        x[d]    = (amc + (z * bmd)) * r3;
                    }
                }

                base = base + (4UL * krange);
            }

            block  = block * 4UL;
            length = length / 4UL;
            krange = krange / 4UL;
            base   = 0;
            tss    = tss * 4;
        }
    }

    [[nodiscard]] static auto make_twiddle_lut(size_type n)
    {
        auto kmax  = 3UL * (n / 4UL - 1UL);
        auto w     = stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>>{2, n};
        auto w_fwd = stdex::submdspan(w.to_mdspan(), 0, stdex::full_extent);
        auto w_bwd = stdex::submdspan(w.to_mdspan(), 1, stdex::full_extent);
        for (auto i{0U}; i < kmax + 1; ++i) {
            w_fwd[i] = twiddle<Complex>(n, i, direction::forward);
            w_bwd[i] = twiddle<Complex>(n, i, direction::backward);
        }
        return w;
    }

    size_type _order;
    digitrevorder_plan<4> _reorder{size()};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>> _w{make_twiddle_lut(size())};
};

template<typename Complex>
using c2c_dit4_plan = c2c_radix4_plan<Complex, true>;

template<typename Complex>
using c2c_dif4_plan = c2c_radix4_plan<Complex, false>;

}  // namespace neo::fft::experimental