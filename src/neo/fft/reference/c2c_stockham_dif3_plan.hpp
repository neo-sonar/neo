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

/// \brief C2C Stockham Radix-3 DIF
///
/// Chapter 3.2 \n
/// Fast Fourier Transform Algorithms for Parallel Computers \n
/// Daisuke Takahashi (2019) \n
/// ISBN 978-981-13-9964-0 \n
/// \ingroup neo-fft
template<complex Complex>
struct c2c_stockham_dif3_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    c2c_stockham_dif3_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{11}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return ipow<size_type(3)>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return ipow<size_type(3)>(order()); }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        using Float = value_type_t<Complex>;

        auto const sign        = dir == direction::forward ? Float(-1) : Float(+1);
        auto const i           = Complex{Float(0), sign};
        auto const sin_pi_by_3 = static_cast<Float>(std::sin(std::numbers::pi / 3.0));

        auto const n = size();
        auto const p = static_cast<size_t>(_order);
        auto const y = _work.to_mdspan();

        auto l = n / 3zu;
        auto m = 1zu;

        for (auto t{1zu}; t <= p; ++t) {
            copy(x, y);

            for (auto j{0zu}; j < l; ++j) {
                auto const w1 = twiddle<Complex>(3 * l, 1 * j, dir);
                auto const w2 = twiddle<Complex>(3 * l, 2 * j, dir);

                for (auto k{0zu}; k < m; ++k) {
                    auto const c0 = y[k + (j * m) + (0 * l * m)];
                    auto const c1 = y[k + (j * m) + (1 * l * m)];
                    auto const c2 = y[k + (j * m) + (2 * l * m)];

                    auto const d0 = c1 + c2;
                    auto const d1 = c0 - d0 * Float(0.5);
                    auto const d2 = (c1 - c2) * i * sin_pi_by_3;

                    x[k + (3 * j * m) + (0 * m)] = c0 + d0;
                    x[k + (3 * j * m) + (1 * m)] = (d1 + d2) * w1;
                    x[k + (3 * j * m) + (2 * m)] = (d1 - d2) * w2;
                }
            }

            l = l / 3;
            m = m * 3;
        }
    }

private:
    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _work{size()};
};

}  // namespace neo::fft
