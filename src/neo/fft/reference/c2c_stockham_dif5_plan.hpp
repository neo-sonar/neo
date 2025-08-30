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

/// \brief C2C Stockham Radix-5 DIF
///
/// Chapter 3.4 \n
/// Fast Fourier Transform Algorithms for Parallel Computers \n
/// Daisuke Takahashi (2019) \n
/// ISBN 978-981-13-9964-0 \n
/// \ingroup neo-fft
template<complex Complex>
struct c2c_stockham_dif5_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    c2c_stockham_dif5_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{11}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return ipow<size_type(5)>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return ipow<size_type(5)>(order()); }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        using Float = value_type_t<Complex>;

        auto const sign = dir == direction::forward ? Float(-1) : Float(+1);
        auto const i    = Complex{Float(0), sign};

        auto const sqrt5_by_4   = static_cast<Float>(std::sqrt(5.0) / 4.0);
        auto const sin_pi_by_5  = static_cast<Float>(std::sin(std::numbers::pi / 5.0));
        auto const sin_2pi_by_5 = static_cast<Float>(std::sin((2.0 * std::numbers::pi) / 5.0));
        auto const sin_ratio    = sin_pi_by_5 / sin_2pi_by_5;

        auto const n = size();
        auto const p = static_cast<size_t>(_order);
        auto const y = _work.to_mdspan();

        auto l = n / 5zu;
        auto m = 1zu;

        for (auto t{1zu}; t <= p; ++t) {
            copy(x, y);

            for (auto j{0zu}; j < l; ++j) {
                auto const w1 = twiddle<Complex>(l * 5, 1 * j, dir);
                auto const w2 = twiddle<Complex>(l * 5, 2 * j, dir);
                auto const w3 = twiddle<Complex>(l * 5, 3 * j, dir);
                auto const w4 = twiddle<Complex>(l * 5, 4 * j, dir);

                for (auto k{0zu}; k < m; ++k) {
                    auto const c0 = y[k + (j * m) + (0 * l * m)];
                    auto const c1 = y[k + (j * m) + (1 * l * m)];
                    auto const c2 = y[k + (j * m) + (2 * l * m)];
                    auto const c3 = y[k + (j * m) + (3 * l * m)];
                    auto const c4 = y[k + (j * m) + (4 * l * m)];

                    auto const d0  = c1 + c4;
                    auto const d1  = c2 + c3;
                    auto const d2  = (c1 - c4) * sin_2pi_by_5;
                    auto const d3  = (c2 - c3) * sin_2pi_by_5;
                    auto const d4  = d0 + d1;
                    auto const d5  = (d0 - d1) * sqrt5_by_4;
                    auto const d6  = c0 - d4 * Float(0.25);
                    auto const d7  = d6 + d5;
                    auto const d8  = d6 - d5;
                    auto const d9  = (d2 + d3 * sin_ratio) * i;
                    auto const d10 = (d2 * sin_ratio - d3) * i;

                    x[k + (5 * j * m) + (0 * m)] = c0 + d4;
                    x[k + (5 * j * m) + (1 * m)] = (d7 + d9) * w1;
                    x[k + (5 * j * m) + (2 * m)] = (d8 + d10) * w2;
                    x[k + (5 * j * m) + (3 * m)] = (d8 - d10) * w3;
                    x[k + (5 * j * m) + (4 * m)] = (d7 - d9) * w4;
                }
            }

            l = l / 5;
            m = m * 5;
        }
    }

private:
    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _work{size()};
};

}  // namespace neo::fft
