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

/// \brief C2C Stockham Radix-8 DIF
///
/// Chapter 3.5 \n
/// Fast Fourier Transform Algorithms for Parallel Computers \n
/// Daisuke Takahashi (2019) \n
/// ISBN 978-981-13-9964-0 \n
/// \ingroup neo-fft
template<complex Complex>
struct c2c_stockham_dif8_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    c2c_stockham_dif8_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{9}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return ipow<size_type(8)>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return ipow<size_type(8)>(order()); }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        using Float = value_type_t<Complex>;

        static constexpr auto sqrt2_by_2 = static_cast<Float>(std::numbers::sqrt2 / 2.0);

        auto const sign = dir == direction::forward ? Float(-1) : Float(+1);
        auto const i    = Complex{Float(0), sign};

        auto const n = size();
        auto const p = static_cast<size_t>(_order);
        auto const y = _work.to_mdspan();

        auto l = n / 8zu;
        auto m = 1zu;

        for (auto t{1zu}; t <= p; ++t) {
            copy(x, y);

            for (auto j{0zu}; j < l; ++j) {
                auto const w1 = twiddle<Complex>(l * 8, 1 * j, dir);
                auto const w2 = twiddle<Complex>(l * 8, 2 * j, dir);
                auto const w3 = twiddle<Complex>(l * 8, 3 * j, dir);
                auto const w4 = twiddle<Complex>(l * 8, 4 * j, dir);
                auto const w5 = twiddle<Complex>(l * 8, 5 * j, dir);
                auto const w6 = twiddle<Complex>(l * 8, 6 * j, dir);
                auto const w7 = twiddle<Complex>(l * 8, 7 * j, dir);

                for (auto k{0zu}; k < m; ++k) {
                    auto const c0 = y[k + (j * m) + (0 * l * m)];
                    auto const c1 = y[k + (j * m) + (1 * l * m)];
                    auto const c2 = y[k + (j * m) + (2 * l * m)];
                    auto const c3 = y[k + (j * m) + (3 * l * m)];
                    auto const c4 = y[k + (j * m) + (4 * l * m)];
                    auto const c5 = y[k + (j * m) + (5 * l * m)];
                    auto const c6 = y[k + (j * m) + (6 * l * m)];
                    auto const c7 = y[k + (j * m) + (7 * l * m)];

                    auto const d0 = c0 + c4;
                    auto const d1 = c0 - c4;
                    auto const d2 = c2 + c6;
                    auto const d3 = (c2 - c6) * i;
                    auto const d4 = c1 + c5;
                    auto const d5 = c1 - c5;
                    auto const d6 = c3 + c7;
                    auto const d7 = c3 - c7;

                    auto const e0 = d0 + d2;
                    auto const e1 = d0 - d2;
                    auto const e2 = d4 + d6;
                    auto const e3 = (d4 - d6) * i;
                    auto const e4 = (d5 - d7) * sqrt2_by_2;
                    auto const e5 = (d5 + d7) * sqrt2_by_2 * i;
                    auto const e6 = d1 + e4;
                    auto const e7 = d1 - e4;
                    auto const e8 = d3 + e5;
                    auto const e9 = d3 - e5;

                    x[k + (8 * j * m) + (m * 0)] = e0 + e2;
                    x[k + (8 * j * m) + (m * 1)] = (e6 + e8) * w1;
                    x[k + (8 * j * m) + (m * 2)] = (e1 + e3) * w2;
                    x[k + (8 * j * m) + (m * 3)] = (e7 - e9) * w3;
                    x[k + (8 * j * m) + (m * 4)] = (e0 - e2) * w4;
                    x[k + (8 * j * m) + (m * 5)] = (e7 + e9) * w5;
                    x[k + (8 * j * m) + (m * 6)] = (e1 - e3) * w6;
                    x[k + (8 * j * m) + (m * 7)] = (e6 - e8) * w7;
                }
            }

            l = l / 8;
            m = m * 8;
        }
    }

private:
    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _work{size()};
};

}  // namespace neo::fft
