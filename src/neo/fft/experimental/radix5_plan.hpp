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

/// https://github.com/scientificgo/fft/blob/master/radix5.go
template<complex Complex>
struct radix5_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit radix5_plan(fft::order order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> fft::order { return fft::order{11}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type
    {
        return ipow<size_type(5)>(static_cast<size_type>(max_order()));
    }

    [[nodiscard]] auto order() const noexcept -> fft::order { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type
    {
        return ipow<size_type(5)>(static_cast<size_type>(order()));
    }

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Complex>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        using Float = value_type_t<Complex>;

        auto const w51 = twiddle<Complex>(5, 1, direction::forward);
        auto const w52 = twiddle<Complex>(5, 2, direction::forward);
        auto const w53 = twiddle<Complex>(5, 3, direction::forward);
        auto const w54 = twiddle<Complex>(5, 4, direction::forward);

        auto const radix = 5;
        auto const len   = static_cast<int>(x.extent(0));

        digitrevorder<5>(x);

        for (auto m = radix; m <= len; m *= radix) {
            // Calculate twiddle factor.
            auto const w = twiddle<Complex>(m, 1, dir);

            auto mr = m / radix;
            auto wi = Complex{Float(1), Float(0)};

            for (auto i = 0; i < mr; i++) {
                for (auto j = 0; j < len; j += m) {
                    // Retrieve subset of points.
                    auto t0 = x[i + j];
                    auto t1 = x[i + j + mr];
                    auto t2 = x[i + j + 2 * mr];
                    auto t3 = x[i + j + 3 * mr];
                    auto t4 = x[i + j + 4 * mr];

                    // Apply twiddle factors w**(i+k) for 1 ≤ k < r.
                    t1 *= wi;
                    t2 *= wi * wi;
                    t3 *= wi * wi * wi;
                    t4 *= wi * wi * wi * wi;

                    // Transform points using r-point DFT.
                    x[i + j] += t1 + t2 + t3 + t4;
                    if (dir == direction::forward) {
                        x[i + j + mr]     = t0 + t1 * w51 + t2 * w52 + t3 * w53 + t4 * w54;
                        x[i + j + 2 * mr] = t0 + t1 * w52 + t2 * w54 + t3 * w51 + t4 * w53;
                        x[i + j + 3 * mr] = t0 + t1 * w53 + t2 * w51 + t3 * w54 + t4 * w52;
                        x[i + j + 4 * mr] = t0 + t1 * w54 + t2 * w53 + t3 * w52 + t4 * w51;

                    } else {
                        // 1/w51 = w54, etc.
                        x[i + j + mr]     = t0 + t1 * w54 + t2 * w53 + t3 * w52 + t4 * w51;
                        x[i + j + 2 * mr] = t0 + t1 * w53 + t2 * w51 + t3 * w54 + t4 * w52;
                        x[i + j + 3 * mr] = t0 + t1 * w52 + t2 * w54 + t3 * w51 + t4 * w53;
                        x[i + j + 4 * mr] = t0 + t1 * w51 + t2 * w52 + t3 * w53 + t4 * w54;
                    }
                }
                wi *= w;
            }
        }
    }

private:
    fft::order _order;
};

}  // namespace neo::fft::experimental