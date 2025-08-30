// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft.hpp>

#include <cassert>
#include <functional>

namespace neo::convolution {

/// \ingroup neo-convolution
template<complex Complex>
struct overlap_save
{
    using value_type   = Complex;
    using complex_type = Complex;
    using real_type    = typename Complex::value_type;
    using size_type    = std::size_t;

    overlap_save(size_type block_size, size_type filter_size);

    [[nodiscard]] auto block_size() const noexcept -> size_type;
    [[nodiscard]] auto filter_size() const noexcept -> size_type;
    [[nodiscard]] auto transform_size() const noexcept -> size_type;

    auto operator()(inout_vector auto block, auto callback) -> void;

private:
    auto slide_window_left(auto window) -> void
    {
        auto const step      = static_cast<int>(block_size());
        auto const num_steps = static_cast<int>(window.extent(0)) / step;
        for (auto i{0}; i < num_steps - 1; ++i) {
            auto const dest_idx = i * step;
            auto const src_idx  = dest_idx + step;

            auto const src_block  = stdex::submdspan(window, std::tuple{src_idx, src_idx + step});
            auto const dest_block = stdex::submdspan(window, std::tuple{dest_idx, src_idx});
            copy(src_block, dest_block);
        }
    }

    size_type _block_size;
    size_type _filter_size;
    fft::rfft_plan<real_type, complex_type> _plan{fft::from_order, fft::next_order(_block_size + _filter_size - 1zu)};

    stdex::mdarray<real_type, stdex::dextents<size_t, 1>> _window{_plan.size()};
    stdex::mdarray<real_type, stdex::dextents<size_t, 1>> _real_buffer{_plan.size()};
    stdex::mdarray<complex_type, stdex::dextents<size_t, 1>> _complex_buffer{_plan.size()};
};

template<complex Complex>
overlap_save<Complex>::overlap_save(size_type block_size, size_type filter_size)
    : _block_size{block_size}
    , _filter_size{filter_size}
{}

template<complex Complex>
auto overlap_save<Complex>::block_size() const noexcept -> size_type
{
    return _block_size;
}

template<complex Complex>
auto overlap_save<Complex>::filter_size() const noexcept -> size_type
{
    return _filter_size;
}

template<complex Complex>
auto overlap_save<Complex>::transform_size() const noexcept -> size_type
{
    return _plan.size();
}

template<complex Complex>
auto overlap_save<Complex>::operator()(inout_vector auto block, auto callback) -> void
{
    assert(block.extent(0) == block_size());

    // Time domain input buffer
    auto const window       = _window.to_mdspan();
    auto const keep_extents = std::tuple{transform_size() - block_size(), transform_size()};

    auto const dest_in_window = stdex::submdspan(window, keep_extents);
    slide_window_left(window);
    copy(block, dest_in_window);

    // 2B-point R2C-FFT
    auto const complex_buf = _complex_buffer.to_mdspan();
    rfft(_plan, window, complex_buf);

    // Apply processing
    auto const coeffs = stdex::submdspan(complex_buf, std::tuple{0zu, _plan.size() / 2zu + 1zu});
    callback(coeffs);

    // 2B-point C2R-IFFT
    auto const real_buf = _real_buffer.to_mdspan();
    irfft(_plan, complex_buf, real_buf);
    scale(real_type(1) / static_cast<real_type>(_plan.size()), real_buf);

    // Copy block_size samples to output
    copy(stdex::submdspan(real_buf, keep_extents), block);
}

}  // namespace neo::convolution
