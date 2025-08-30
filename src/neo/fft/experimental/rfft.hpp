// SPDX-License-Identifier: MIT

#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/reference/bitrevorder.hpp>
#include <neo/fft/reference/c2c_dit2_plan.hpp>
#include <neo/fft/twiddle.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

namespace neo::fft::experimental {

namespace detail {

struct c2c_kernel
{
    c2c_kernel() = default;

    template<inout_vector Vec>
        requires std::floating_point<typename Vec::value_type>
    auto operator()(Vec x, auto const& twiddles) const noexcept -> void
    {
        auto const size  = static_cast<int>(x.extent(0) / 2);
        auto const order = static_cast<int>(bit_log2(static_cast<unsigned>(size)));

        {
            // stage 0
            static constexpr auto const stage_length = 1;  // ipow<2>(0)
            static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

            for (auto k{0}; k < static_cast<int>(size); k += stride) {
                auto const i1 = k;
                auto const i2 = k + stage_length;

                auto const i1re = i1 * 2;
                auto const i1im = i1re + 1;

                auto const i2re = i2 * 2;
                auto const i2im = i2re + 1;

                auto const x1re = x[i1re];
                auto const x1im = x[i1im];

                auto const x2re = x[i2re];
                auto const x2im = x[i2im];

                x[i1re] = x1re + x2re;
                x[i1im] = x1im + x2im;

                x[i2re] = x1re - x2re;
                x[i2im] = x1im - x2im;
            }
        }

        auto stage_length = 2;
        auto stride       = 4;

        for (auto stage{1}; stage < order; ++stage) {
            auto const tw_stride = ipow<2>(order - stage - 1);

            for (auto k{0}; k < size; k += stride) {
                for (auto pair{0}; pair < stage_length; ++pair) {
                    auto const tw = twiddles[pair * tw_stride];

                    auto const i1   = k + pair;
                    auto const i1re = i1 * 2;
                    auto const i1im = i1re + 1;

                    auto const i2   = i1 + stage_length;
                    auto const i2re = i2 * 2;
                    auto const i2im = i2re + 1;

                    auto const x1 = std::complex{x[i1re], x[i1im]};
                    auto const x2 = std::complex{x[i2re], x[i2im]};

                    auto const xn1 = x1 + tw * x2;
                    auto const xn2 = x1 - tw * x2;

                    x[i1re] = xn1.real();
                    x[i1im] = xn1.imag();

                    x[i2re] = xn2.real();
                    x[i2im] = xn2.imag();
                }
            }

            stage_length *= 2;
            stride *= 2;
        }
    }
};

}  // namespace detail

template<std::floating_point Float>
struct c2c_dit2_plan
{
    using value_type = Float;
    using size_type  = std::size_t;

    c2c_dit2_plan(from_order_tag /*tag*/, size_type order)
        : _order{order}
        , _wf{neo::fft::make_twiddle_lut_radix2<std::complex<Float>>(size(), direction::forward)}
        , _wb{neo::fft::make_twiddle_lut_radix2<std::complex<Float>>(size(), direction::backward)}
    {}

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return size_type(1) << _order; }

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Float>
    auto operator()(Vec x, direction dir) -> void
    {
        assert(std::cmp_equal(x.extent(0) / 2zu, size()));

        _rev(x);

        if (dir == direction::forward) {
            detail::c2c_kernel{}(x, _wf.to_mdspan());
        } else {
            detail::c2c_kernel{}(x, _wb.to_mdspan());
        }
    }

private:
    size_type _order;
    bitrevorder_plan _rev{_order};
    stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 1>> _wf;
    stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 1>> _wb;
};

template<std::floating_point Float>
struct rfft_plan
{
    using value_type = Float;
    using size_type  = std::size_t;

    rfft_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return size_type(1) << _order; }

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Float>
    auto operator()(Vec x, direction dir) -> void
    {
        auto const n     = x.extent(0);
        auto const c1    = Float(0.5);
        auto const c2    = dir == direction::forward ? Float(-0.5) : Float(0.5);
        auto const theta = [=] {
            auto const t = static_cast<Float>(std::numbers::pi) / static_cast<Float>(n >> 1);
            return dir == direction::forward ? t : -t;
        }();

        if (dir == direction::forward) {
            _fft(x, direction::forward);
        }

        auto wtemp     = std::sin(Float(0.5) * theta);
        auto const wpr = Float(-2) * wtemp * wtemp;
        auto const wpi = std::sin(theta);
        auto wr        = Float(1) + wpr;
        auto wi        = wpi;

        for (auto i = 1zu; i < (n >> 2zu); i++) {
            auto const i1 = i + i;
            auto const i2 = i1 + 1;
            auto const i3 = n - i1;
            auto const i4 = i3 + 1;

            auto const h1r = c1 * (x[i1] + x[i3]);
            auto const h1i = c1 * (x[i2] - x[i4]);
            auto const h2r = -c2 * (x[i2] + x[i4]);
            auto const h2i = c2 * (x[i1] - x[i3]);

            x[i1] = h1r + wr * h2r - wi * h2i;
            x[i2] = h1i + wr * h2i + wi * h2r;
            x[i3] = h1r - wr * h2r + wi * h2i;
            x[i4] = -h1i + wr * h2i + wi * h2r;

            auto const tmp = wr;
            wr             = tmp * wpr - wi * wpi + wr;
            wi             = wi * wpr + tmp * wpi + wi;
        }

        auto const h1r = x[0];
        if (dir == direction::forward) {
            x[0] = h1r + x[1];
            x[1] = h1r - x[1];
        } else {
            x[0] = c1 * (h1r + x[1]);
            x[1] = c1 * (h1r - x[1]);
            _fft(x, direction::backward);
        }
    }

private:
    size_type _order;
    c2c_dit2_plan<Float> _fft{from_order, _order - 1zu};
};

}  // namespace neo::fft::experimental
