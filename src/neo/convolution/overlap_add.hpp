// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/add.hpp>
#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/convolution/mode.hpp>
#include <neo/fft/rfft.hpp>
#include <neo/math/idiv.hpp>

#include <cassert>
#include <functional>

namespace neo::convolution {

/// \ingroup neo-convolution
template<complex Complex>
struct overlap_add
{
    using value_type   = Complex;
    using complex_type = Complex;
    using real_type    = typename Complex::value_type;
    using size_type    = std::size_t;

    overlap_add(size_type block_size, size_type filter_size);

    [[nodiscard]] auto block_size() const noexcept -> size_type;
    [[nodiscard]] auto filter_size() const noexcept -> size_type;
    [[nodiscard]] auto transform_size() const noexcept -> size_type;

    auto operator()(inout_vector auto block, auto callback) -> void;

private:
    size_type _block_size;
    size_type _filter_size;

    fft::rfft_plan<real_type, complex_type> _rfft{
        fft::from_order,
        fft::next_order(output_size<mode::full>(_block_size, _filter_size)),
    };

    stdex::mdarray<real_type, stdex::dextents<size_t, 1>> _window{_rfft.size()};
    stdex::mdarray<complex_type, stdex::dextents<size_t, 1>> _spectrum{_rfft.size() / 2 + 1};
    stdex::mdarray<real_type, stdex::dextents<size_t, 1>> _overlap{_block_size};
};

template<complex Complex>
overlap_add<Complex>::overlap_add(size_type block_size, size_type filter_size)
    : _block_size{block_size}
    , _filter_size{filter_size}
{}

template<complex Complex>
auto overlap_add<Complex>::block_size() const noexcept -> size_type
{
    return _block_size;
}

template<complex Complex>
auto overlap_add<Complex>::filter_size() const noexcept -> size_type
{
    return _filter_size;
}

template<complex Complex>
auto overlap_add<Complex>::transform_size() const noexcept -> size_type
{
    return _rfft.size();
}

template<complex Complex>
auto overlap_add<Complex>::operator()(inout_vector auto block, auto callback) -> void
{
    assert(block.extent(0) == block_size());

    auto const window   = _window.to_mdspan();
    auto const signal   = stdex::submdspan(window, std::tuple{0, block_size()});
    auto const padding  = stdex::submdspan(window, std::tuple{block_size(), block_size() * 2U});
    auto const spectrum = _spectrum.to_mdspan();
    auto const overlap  = _overlap.to_mdspan();

    // Copy and zero-pad input
    copy(block, signal);
    fill(padding, real_type(0));

    // K-point rfft
    rfft(_rfft, window, spectrum);

    // Convolve
    callback(spectrum);

    // K-point irfft
    irfft(_rfft, spectrum, window);
    scale(real_type(1) / static_cast<real_type>(_rfft.size()), window);

    // Copy to output
    add(signal, overlap, block);

    // Save overlap for next iteration
    copy(padding, overlap);
}

}  // namespace neo::convolution
