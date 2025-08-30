// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/order.hpp>
#include <neo/fft/twiddle.hpp>
#include <neo/math/conj.hpp>
#include <neo/math/ipow.hpp>

namespace neo::fft {

/// \brief C2C Stockham Radix-4 DIT
/// \ingroup neo-fft
template<complex Complex>
struct c2c_stockham_dit4_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    c2c_stockham_dit4_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{11}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return ipow<size_type(4)>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return ipow<size_type(4)>(order()); }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        using Float = value_type_t<Complex>;

        auto const n = size();
        auto const t = static_cast<size_t>(_order);
        auto const y = _work.to_mdspan();

        auto const sign = dir == direction::forward ? Float(-1) : Float(+1);
        auto const i    = Complex{Float(0), sign};
        // auto const w    = dir == direction::forward
        //                     ? stdex::submdspan(_w.to_mdspan(), 0, stdex::full_extent, stdex::full_extent)
        //                     : stdex::submdspan(_w.to_mdspan(), 1, stdex::full_extent, stdex::full_extent);

        for (auto q{1zu}; q <= t; ++q) {
            auto const l     = ipow<4zu>(q);
            auto const r     = n / l;
            auto const lstar = l / 4zu;
            auto const rstar = 4zu * r;

            copy(x, y);

            for (auto j{0zu}; j < lstar; ++j) {
                auto const w1 = twiddle<Complex>(l, 1 * j, dir);
                auto const w2 = twiddle<Complex>(l, 2 * j, dir);
                auto const w3 = twiddle<Complex>(l, 3 * j, dir);

                // auto const w1 = twiddle<Complex>(n / 4, 1 * j * r, dir);
                // auto const w2 = twiddle<Complex>(n / 4, 2 * j * r, dir);
                // auto const w3 = twiddle<Complex>(n / 4, 3 * j * r, dir);

                // auto const v  = (ipow<4zu>(q - 1zu) - 1zu) / 3;
                // auto const vj = v + j;
                // auto const w1 = w(vj, 0);
                // auto const w2 = w(vj, 1);
                // auto const w3 = w(vj, 2);

                for (auto k{0zu}; k < r; ++k) {
                    auto const a = y[j * rstar + r * 0 + k];
                    auto const b = y[j * rstar + r * 1 + k] * w1;
                    auto const c = y[j * rstar + r * 2 + k] * w2;
                    auto const d = y[j * rstar + r * 3 + k] * w3;

                    auto const t0 = a + c;
                    auto const t1 = a - c;
                    auto const t2 = b + d;
                    auto const t3 = b - d;

                    x[(j + lstar * 0) * r + k] = t0 + t2;
                    x[(j + lstar * 1) * r + k] = t1 - t3 * i;
                    x[(j + lstar * 2) * r + k] = t0 - t2;
                    x[(j + lstar * 3) * r + k] = t1 + t3 * i;
                }
            }
        }
    }

private:
    static auto make_twiddle_lut(size_t n) -> stdex::mdarray<Complex, stdex::dextents<std::size_t, 3>>
    {
        auto w = stdex::mdarray<Complex, stdex::dextents<std::size_t, 3>>(2zu, n / 4zu, 3zu);
        for (std::size_t i = 0; i < w.size(); i++) {
            // w(0, i, 0) = twiddle<Complex>(n, 1 * i, fft::direction::forward);
            // w(0, i, 1) = twiddle<Complex>(n, 2 * i, fft::direction::forward);
            // w(0, i, 2) = twiddle<Complex>(n, 3 * i, fft::direction::forward);

            // w(0, i, 0) = twiddle<Complex>(n, 1 * i, fft::direction::backward);
            // w(0, i, 1) = twiddle<Complex>(n, 2 * i, fft::direction::backward);
            // w(0, i, 2) = twiddle<Complex>(n, 3 * i, fft::direction::backward);
        }
        return w;
    }

    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 3>> _w{make_twiddle_lut(size())};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _work{size()};
};

}  // namespace neo::fft
