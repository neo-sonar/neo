// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/order.hpp>
#include <neo/fft/twiddle.hpp>
#include <neo/math/conj.hpp>
#include <neo/math/ipow.hpp>
#include <neo/math/is_even.hpp>
#include <neo/math/is_odd.hpp>

#include <cstdint>

namespace neo::fft {

/// \brief C2C Stockham Radix-2 DIF (Recursive)
/// \details http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham2.html
/// \ingroup neo-fft
template<complex Complex>
struct c2c_stockham_dif2r_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    c2c_stockham_dif2r_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{20}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return ipow<size_type(2)>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return ipow<size_type(2)>(order()); }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        if (dir == direction::forward) {
            return fft(x);
        }
        return ifft(x);
    }

private:
    auto fft(inout_vector_of<Complex> auto x) noexcept -> void
    {
        fft0(size(), 1, false, x, _work.to_mdspan(), _w.to_mdspan());
    }

    void ifft(inout_vector_of<Complex> auto x)
    {
        auto conjugate = [n = size()](inout_vector_of<Complex> auto vec) {
            for (auto i{0zu}; i < n; ++i) {
                vec[i] = neo::math::conj(vec[i]);
            }
        };

        conjugate(x);
        fft0(size(), 1, false, x, _work.to_mdspan(), _w.to_mdspan());
        conjugate(x);
    }

    // stage_len:      sequence length
    // s:              stride
    // work_is_output: x is output if work_is_output == false, work is output if work_is_output == true
    // x:              input sequence(or output sequence if work_is_output == false)
    // work:           work area(or output sequence if work_is_output == true)
    static auto fft0(
        std::size_t stage_len,
        std::size_t stride,
        bool work_is_output,
        inout_vector_of<Complex> auto x,
        inout_vector_of<Complex> auto work,
        in_vector_of<Complex> auto w
    ) -> void
    {
        if (stage_len == 1) {
            if (work_is_output) {
                for (std::size_t k = 0; k < stride; k++) {
                    work[k] = x[k];
                }
            }
            return;
        }

        auto const m = stage_len / 2zu;

        for (std::size_t j = 0; j < m; j++) {
            auto const w1 = w[j * stride];

            for (std::size_t k = 0; k < stride; k++) {
                auto const a = x[k + stride * (j + 0)];
                auto const b = x[k + stride * (j + m)];

                work[k + stride * (2 * j + 0)] = a + b;
                work[k + stride * (2 * j + 1)] = (a - b) * w1;
            }
        }

        fft0(stage_len / 2, stride * 2, !work_is_output, work, x, w);
    }

    static auto make_twiddle_lut(size_t n) -> stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>>
    {
        auto w = stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>>(n / 2);
        for (auto i{0zu}; i < w.size(); i++) {
            w(i) = twiddle<Complex>(n, i, fft::direction::backward);
        }
        return w;
    }

    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _w{make_twiddle_lut(size())};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _work{size()};
};

/// \brief C2C Stockham Radix-2 DIF (Iterative)
///
/// Chapter 2.5 \n
/// Fast Fourier Transform Algorithms for Parallel Computers \n
/// Daisuke Takahashi (2019) \n
/// ISBN 978-981-13-9964-0 \n
/// \ingroup neo-fft
template<complex Complex>
struct c2c_stockham_dif2i_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    c2c_stockham_dif2i_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{20}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return ipow<size_type(2)>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return ipow<size_type(2)>(order()); }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        auto const n = static_cast<int>(size());
        auto const y = _work.to_mdspan();
        auto const w = dir == direction::forward ? stdex::submdspan(_w.to_mdspan(), 0, stdex::full_extent)
                                                 : stdex::submdspan(_w.to_mdspan(), 1, stdex::full_extent);

        for (auto stage{0}; stage < static_cast<int>(order()); ++stage) {
            auto const stride    = ipow<2>(stage);
            auto const stage_len = n / (stride * 2);
            auto const offset    = stage_len * stride;

            if (is_even(stage)) {
                butterfly(x, y, w, stage_len, stride, offset);
            } else {
                butterfly(y, x, w, stage_len, stride, offset);
            }
        }

        if (is_odd(order())) {
            copy(y, x);
        }
    }

private:
    auto butterfly(
        in_vector_of<Complex> auto x,
        out_vector_of<Complex> auto y,
        in_vector_of<Complex> auto w,
        std::int32_t stage_len,
        std::int32_t stride,
        std::int32_t offset
    ) noexcept -> void
    {
        for (auto j{0}; j < stage_len; ++j) {
            auto const jm  = j * stride;
            auto const jm2 = jm * 2;
            auto const w1  = w[jm];

            for (auto k{0}; k < stride; ++k) {
                auto const c0 = x[k + jm];
                auto const c1 = x[k + jm + offset];

                y[k + jm2]          = c0 + c1;
                y[k + jm2 + stride] = (c0 - c1) * w1;
            }
        }
    }

    static auto make_twiddle_lut(size_t n) -> stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>>
    {
        auto w = stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>>{2, n / 2};
        for (auto i{0zu}; i < w.extent(1); i++) {
            w(0, i) = twiddle<Complex>(n, i, fft::direction::backward);
            w(1, i) = twiddle<Complex>(n, i, fft::direction::forward);
        }
        return w;
    }

    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>> _w{make_twiddle_lut(size())};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _work{size()};
};

}  // namespace neo::fft
