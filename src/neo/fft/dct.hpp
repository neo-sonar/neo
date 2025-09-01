// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fft.hpp>

#if defined(NEO_HAS_INTEL_IPP)
    #include <neo/fft/backend/ipp.hpp>
#endif

#include <cassert>
#include <concepts>
#include <cstddef>

namespace neo::fft {

/// Type 2 DCT using N FFT (Makhoul)
///
/// https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft/10606#10606
/// \ingroup neo-fft
template<std::floating_point Float>
struct fallback_dct2_plan
{
    using value_type = Float;
    using size_type  = std::size_t;

    fallback_dct2_plan(from_order_tag tag, size_type order) : _fft{tag, order} {}

    [[nodiscard]] auto order() const noexcept -> size_type { return _fft.order(); }

    [[nodiscard]] auto size() const noexcept -> size_type { return _fft.size(); }

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Float>
    auto operator()(Vec x) noexcept -> void
    {
        auto const len = x.extent(0);
        auto const buf = _buffer.to_mdspan();
        assert(len == size());

        for (auto i{0zu}; i < len / 2zu; ++i) {
            auto const src = i * 2zu;
            auto const top = len - i - 1;

            buf[i]   = x[src];
            buf[top] = x[src + 1];
        }

        _fft(buf, direction::forward);

        auto const pi  = static_cast<Float>(std::numbers::pi);
        auto const n   = static_cast<Float>(len);
        auto const one = std::complex{Float(0), Float(-1)};

        for (auto i{0zu}; i < len; ++i) {
            auto const scale = Float(2) * std::exp(one * pi * Float(i) / (2 * n));
            auto const v     = buf[i] * scale;

            x[i] = v.real();
        }
    }

private:
    fft_plan<std::complex<Float>> _fft;
    stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 1>> _buffer{size()};
};

}  // namespace neo::fft
