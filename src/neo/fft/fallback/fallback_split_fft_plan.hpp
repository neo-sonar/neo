// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fft.hpp>
#include <neo/fft/order.hpp>
#include <neo/math/imag.hpp>
#include <neo/math/real.hpp>

namespace neo::fft {

/// \ingroup neo-fft
template<std::floating_point Float>
struct fallback_split_fft_plan
{
    using value_type = Float;
    using size_type  = std::size_t;

    fallback_split_fft_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<inout_vector_of<Float> InOutVec>
    auto operator()(split_complex<InOutVec> x, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.real.extent(0), size()));
        assert(neo::detail::extents_equal(x.real, x.imag));

        _reorder(x);

        if (dir == direction::forward) {
            stage_0(x.real, x.imag);
        } else {
            stage_0(x.imag, x.real);
        }
    }

    template<in_vector_of<Float> InVec, out_vector_of<Float> OutVec>
    auto operator()(split_complex<InVec> in, split_complex<OutVec> out, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(in.real.extent(0), size()));
        assert(neo::detail::extents_equal(in.real, in.imag, out.real, out.imag));

        copy(in.real, out.real);
        copy(in.imag, out.imag);
        (*this)(out, dir);
    }

private:
    auto stage_0(inout_vector_of<Float> auto xre, inout_vector_of<Float> auto xim) -> void
    {
        static constexpr auto const stage_length = 1;  // ipow<2>(0)
        static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

        for (auto k{0}; k < static_cast<int>(size()); k += stride) {
            auto const i1 = k;
            auto const i2 = k + stage_length;

            auto const x1re = xre[i1];
            auto const x1im = xim[i1];
            auto const x2re = xre[i2];
            auto const x2im = xim[i2];

            xre[i1] = x1re + x2re;
            xim[i1] = x1im + x2im;

            xre[i2] = x1re - x2re;
            xim[i2] = x1im - x2im;
        }

        auto const tw_re = stdex::submdspan(_tw.to_mdspan(), 0, stdex::full_extent);
        auto const tw_im = stdex::submdspan(_tw.to_mdspan(), 1, stdex::full_extent);
        stage_n(xre, xim, tw_re, tw_im);
    }

    auto stage_n(
        inout_vector_of<Float> auto xre,
        inout_vector_of<Float> auto xim,
        in_vector_of<Float> auto tw_re,
        in_vector_of<Float> auto tw_im
    ) -> void
    {
        auto const log2_size = static_cast<int>(order());
        auto const size      = 1 << log2_size;

        for (auto stage{1}; stage < log2_size; ++stage) {

            auto const stage_length = ipow<2>(stage);
            auto const stride       = ipow<2>(stage + 1);
            auto const tw_stride    = ipow<2>(log2_size - stage - 1);

            for (auto k{0}; k < size; k += stride) {
                for (auto pair{0}; pair < stage_length; ++pair) {
                    auto const i1      = k + pair;
                    auto const i2      = k + pair + stage_length;
                    auto const w_index = pair * tw_stride;

                    auto const wre = tw_re[w_index];
                    auto const wim = tw_im[w_index];

                    auto const x1re = xre[i1];
                    auto const x1im = xim[i1];
                    auto const x2re = xre[i2];
                    auto const x2im = xim[i2];

                    auto const xwre = wre * x2re - wim * x2im;
                    auto const xwim = wre * x2im + wim * x2re;

                    xre[i1] = x1re + xwre;
                    xim[i1] = x1im + xwim;
                    xre[i2] = x1re - xwre;
                    xim[i2] = x1im - xwim;
                }
            }
        }
    }

    [[nodiscard]] static auto make_twiddles(size_type n)
    {
        auto interleaved = make_twiddle_lut_radix2<std::complex<Float>>(n, direction::forward);

        auto w_buf = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, interleaved.extent(0)};
        auto w_re  = stdex::submdspan(w_buf.to_mdspan(), 0, stdex::full_extent);
        auto w_im  = stdex::submdspan(w_buf.to_mdspan(), 1, stdex::full_extent);

        copy(interleaved.to_mdspan(), split_complex{w_re, w_im});
        return w_buf;
    }

    size_type _order;
    bitrevorder_plan _reorder{static_cast<size_t>(_order)};
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _tw{make_twiddles(size())};
};

}  // namespace neo::fft
