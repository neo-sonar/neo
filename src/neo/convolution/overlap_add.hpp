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
    [[nodiscard]] auto num_overlaps() const noexcept -> size_type;

    auto operator()(inout_vector auto block, auto callback) -> void;

private:
    size_type _block_size;
    size_type _filter_size;
    size_type _usable_coeffs{output_size<mode::full>(_block_size, _filter_size)};

    fft::rfft_plan<real_type, complex_type> _rfft{fft::next_order(_usable_coeffs)};
    stdex::mdarray<real_type, stdex::dextents<size_t, 1>> _real_buffer{_rfft.size()};
    stdex::mdarray<complex_type, stdex::dextents<size_t, 1>> _complex_buffer{_rfft.size()};

    size_type _overlap_write_idx{0};
    stdex::mdarray<real_type, stdex::dextents<size_t, 2>> _overlaps;
};

template<complex Complex>
overlap_add<Complex>::overlap_add(size_type block_size, size_type filter_size)
    : _block_size{block_size}
    , _filter_size{filter_size}
    , _overlaps{idiv(_usable_coeffs, _block_size), _usable_coeffs}
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
auto overlap_add<Complex>::num_overlaps() const noexcept -> size_type
{
    return _overlaps.extent(0);
}

template<complex Complex>
auto overlap_add<Complex>::operator()(inout_vector auto block, auto callback) -> void
{
    assert(block.extent(0) == block_size());

    auto const real_buffer    = _real_buffer.to_mdspan();
    auto const complex_buffer = _complex_buffer.to_mdspan();

    // Copy and zero-pad input
    fill(real_buffer, real_type(0));
    copy(block, stdex::submdspan(real_buffer, std::tuple{0, block_size()}));

    // K-point rfft
    _rfft(real_buffer, complex_buffer);

    // Process
    auto const coeffs = stdex::submdspan(complex_buffer, std::tuple{0, _rfft.size() / 2 + 1});
    callback(coeffs);

    // K-point irfft
    _rfft(complex_buffer, real_buffer);
    scale(1.0F / static_cast<real_type>(_rfft.size()), real_buffer);

    // Copy to output
    copy(stdex::submdspan(real_buffer, std::tuple{0, block_size()}), block);

    // Save overlap for next iteration
    auto const signal = stdex::submdspan(real_buffer, std::tuple{0, _usable_coeffs});
    auto const save   = stdex::submdspan(_overlaps.to_mdspan(), _overlap_write_idx, stdex::full_extent);
    copy(signal, save);

    // Add older iterations
    for (auto i{1UL}; i < num_overlaps(); ++i) {
        auto const overlap_idx = (_overlap_write_idx + num_overlaps() - i) % num_overlaps();
        auto const start_idx   = block_size() * i;
        auto const num_samples = std::min(block_size(), _usable_coeffs - start_idx);
        auto const last_idx    = start_idx + num_samples;

        auto const overlap = stdex::submdspan(_overlaps.to_mdspan(), overlap_idx, std::tuple{start_idx, last_idx});
        auto const out     = stdex::submdspan(block, std::tuple{0, num_samples});
        add(overlap, out, out);
    }

    // Increment overlap write position
    ++_overlap_write_idx;
    if (_overlap_write_idx == num_overlaps()) {
        _overlap_write_idx = 0;
    }
}

}  // namespace neo::convolution
