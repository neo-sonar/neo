// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/multiply.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/rfft.hpp>
#include <neo/math/idiv.hpp>
#include <neo/math/windowing.hpp>

#include <functional>
#include <utility>

namespace neo::fft {

namespace detail {

[[nodiscard]] constexpr auto
num_sftf_frames(std::size_t signal_size, std::size_t frame_size, std::size_t overlap_size) noexcept
{
    return idiv(signal_size - frame_size + overlap_size, frame_size - overlap_size) + 1;
}

}  // namespace detail

/// \ingroup neo-fft
template<std::floating_point Float>
struct stft_options
{
    std::size_t frame_size{};
    std::size_t transform_size{};
    std::size_t overlap_size{};
    std::function<Float(std::size_t, std::size_t)> window{hann_window<Float>{}};
};

/// \ingroup neo-fft
template<std::floating_point Float, complex Complex = std::complex<Float>>
struct stft_plan
{
    explicit stft_plan(std::size_t transform_size)
        : stft_plan({
              .frame_size     = transform_size,
              .transform_size = transform_size,
              .overlap_size   = transform_size / 2,
          })
    {}

    explicit stft_plan(stft_options<Float> options) : _options{std::move(options)}
    {
        fill_window(_window.to_mdspan(), _options.window);
    }

    template<in_matrix InMat>
        requires std::convertible_to<value_type_t<InMat>, Float>
    [[nodiscard]] auto operator()(InMat x) -> stdex::mdarray<Complex, stdex::dextents<std::size_t, 3>>
    {
        auto const frame_len = _options.frame_size;
        auto const overlap   = _options.overlap_size;

        auto const num_channels = static_cast<std::size_t>(x.extent(0));
        auto const signal_len   = static_cast<std::size_t>(x.extent(1));

        auto const num_bins   = _rfft.size() / 2zu + 1zu;
        auto const num_frames = detail::num_sftf_frames(signal_len, frame_len, overlap);

        auto const in  = _input.to_mdspan();
        auto const out = _output.to_mdspan();

        auto result = stdex::mdarray<Complex, stdex::dextents<std::size_t, 3>>{
            num_channels,
            num_frames,
            num_bins,
        };

        for (auto ch_idx = std::size_t(0); ch_idx < num_channels; ++ch_idx) {
            for (auto frame_idx = std::size_t(0); frame_idx < num_frames; ++frame_idx) {
                auto const sample_idx  = frame_idx * frame_len - frame_idx * overlap;
                auto const num_samples = std::min(signal_len - sample_idx, frame_len);

                auto const in_block  = stdex::submdspan(x, ch_idx, std::tuple{sample_idx, sample_idx + num_samples});
                auto const in_window = stdex::submdspan(in, std::tuple{0, num_samples});
                fill(in, Float(0));
                fill(out, Float(0));
                copy(in_block, in_window);

                multiply(in, _window.to_mdspan(), in);
                _rfft(in, out);

                auto coeffs       = stdex::submdspan(out, std::tuple{0, num_bins});
                auto result_frame = stdex::submdspan(result.to_mdspan(), ch_idx, frame_idx, std::tuple{0, num_bins});
                copy(coeffs, result_frame);
            }
        }

        return result;
    }

private:
    stft_options<Float> _options;

    rfft_plan<Float> _rfft{from_order, next_order(_options.transform_size)};
    stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> _input{_rfft.size()};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _output{_rfft.size()};

    stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> _window{_rfft.size()};
};

/// \ingroup neo-fft
template<in_matrix InMat>
[[nodiscard]] auto stft(InMat x, stft_options<typename InMat::value_type> options)
{
    auto plan = stft_plan<typename InMat::value_type>{std::move(options)};
    return plan(x);
}

/// \ingroup neo-fft
template<in_matrix InMat>
[[nodiscard]] auto stft(InMat x, std::size_t window_size)
{
    auto plan = stft_plan<typename InMat::value_type>{window_size};
    return plan(x);
}

}  // namespace neo::fft
