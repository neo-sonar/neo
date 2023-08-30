#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft.hpp>
#include <neo/math/bit_ceil.hpp>
#include <neo/math/ilog2.hpp>

#include <cassert>
#include <functional>

namespace neo::fft {

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
    size_type _block_size;
    size_type _filter_size;
    rfft_radix2_plan<real_type> _rfft{ilog2(bit_ceil(_block_size + _filter_size - 1UL))};

    stdex::mdarray<real_type, stdex::dextents<size_t, 1>> _window{_rfft.size()};
    stdex::mdarray<real_type, stdex::dextents<size_t, 1>> _real_buffer{_rfft.size()};
    stdex::mdarray<complex_type, stdex::dextents<size_t, 1>> _complex_buffer{_rfft.size()};
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
    return _rfft.size();
}

template<complex Complex>
auto overlap_save<Complex>::operator()(inout_vector auto block, auto callback) -> void
{
    assert(block.extent(0) == block_size());

    auto slide_window_left = [step = static_cast<int>(block_size())](auto window) {
        auto const num_steps = static_cast<int>(window.extent(0)) / step;
        for (auto i{0}; i < num_steps - 1; ++i) {
            auto const dest_idx = i * step;
            auto const src_idx  = dest_idx + step;

            auto const src_block  = stdex::submdspan(window, std::tuple{src_idx, src_idx + step});
            auto const dest_block = stdex::submdspan(window, std::tuple{dest_idx, src_idx});
            copy(src_block, dest_block);
        }
    };

    // Time domain input buffer
    auto const window       = _window.to_mdspan();
    auto const keep_extents = std::tuple{transform_size() - block_size(), transform_size()};

    auto const dest_in_window = stdex::submdspan(window, keep_extents);
    slide_window_left(window);
    copy(block, dest_in_window);

    // 2B-point R2C-FFT
    auto const complex_buf = _complex_buffer.to_mdspan();
    _rfft(window, complex_buf);

    // Scale R2C output
    auto const num_coeffs = _rfft.size() / 2 + 1;
    auto const coeffs     = stdex::submdspan(complex_buf, std::tuple{0, num_coeffs});
    scale(1.0F / static_cast<real_type>(_rfft.size()), coeffs);

    // Apply processing
    callback(coeffs);

    // 2B-point C2R-IFFT
    auto const real_buf = _real_buffer.to_mdspan();
    _rfft(complex_buf, real_buf);

    // Copy block_size samples to output
    auto out = stdex::submdspan(real_buf, keep_extents);
    copy(out, block);
}

}  // namespace neo::fft