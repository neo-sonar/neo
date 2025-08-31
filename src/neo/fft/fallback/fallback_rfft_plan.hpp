// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fft.hpp>
#include <neo/fft/order.hpp>
#include <neo/math/conj.hpp>

namespace neo::fft {

/// \ingroup neo-fft
template<typename Float, typename Complex = std::complex<Float>>
struct fallback_rfft_plan
{
    using real_type    = Float;
    using complex_type = Complex;
    using size_type    = std::size_t;

    fallback_rfft_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<in_vector_of<Float> InVec, out_vector_of<Complex> OutVec>
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        auto const buf    = _buffer.to_mdspan();
        auto const coeffs = size() / 2 + 1;

        copy(in, buf);
        _fft(buf, direction::forward);
        copy(stdex::submdspan(buf, std::tuple{0zu, coeffs}), stdex::submdspan(out, std::tuple{0zu, coeffs}));
    }

    template<in_vector_of<Complex> InVec, out_vector_of<Float> OutVec>
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        auto const buf    = _buffer.to_mdspan();
        auto const coeffs = size() / 2 + 1;

        copy(in, stdex::submdspan(buf, std::tuple{0, in.extent(0)}));

        // Fill upper half with conjugate
        for (auto i{coeffs}; i < size(); ++i) {
            buf[i] = math::conj(buf[size() - i]);
        }

        _fft(buf, direction::backward);
        for (auto i{0zu}; i < size(); ++i) {
            out[i] = buf[i].real();
        }
    }

private:
    size_type _order;
    fft_plan<Complex> _fft{from_order, _order};
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _buffer{size()};
};

}  // namespace neo::fft
