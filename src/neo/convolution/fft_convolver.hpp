// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/multiply.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/convolution/mode.hpp>
#include <neo/fft/rfft.hpp>

#include <cassert>
#include <utility>

namespace neo::convolution {

/// \ingroup neo-convolution
template<std::floating_point Float>
struct fft_convolver
{
    fft_convolver(std::size_t signal_size, std::size_t patch_size) : _signal_size{signal_size}, _patch_size{patch_size}
    {
        assert(_signal_size > 0);
        assert(_patch_size > 0);
    }

    [[nodiscard]] auto signal_size() const noexcept -> std::size_t { return _signal_size; }

    [[nodiscard]] auto patch_size() const noexcept -> std::size_t { return _patch_size; }

    [[nodiscard]] auto output_size() const noexcept -> std::size_t
    {
        return convolution::output_size<mode::full>(signal_size(), patch_size());
    }

    template<in_vector_of<Float> Signal, in_vector_of<Float> Patch, out_vector_of<Float> Output>
    auto operator()(Signal signal, Patch patch, Output output) -> void
    {
        assert(signal.extent(0) == signal_size());
        assert(patch.extent(0) == patch_size());
        assert(output.extent(0) == output_size());

        auto const signal_spectrum = _signal_spectrum.to_mdspan();
        auto const patch_spectrum  = _patch_spectrum.to_mdspan();

        zero_pad_and_transform_forward(signal, signal_spectrum);
        zero_pad_and_transform_forward(patch, patch_spectrum);
        multiply(signal_spectrum, patch_spectrum, signal_spectrum);
        transform_backward(signal_spectrum, output);
    }

private:
    auto zero_pad_and_transform_forward(in_vector auto in, out_vector auto out)
    {
        auto const tmp = _tmp.to_mdspan();
        copy(in, stdex::submdspan(tmp, std::tuple{0, in.extent(0)}));
        fill(stdex::submdspan(tmp, std::tuple{in.extent(0), tmp.extent(0)}), Float(0));
        rfft(_plan, tmp, out);
    }

    auto transform_backward(in_vector auto in, out_vector auto out)
    {
        auto const tmp = _tmp.to_mdspan();
        irfft(_plan, in, tmp);
        scale(Float(1) / Float(_plan.size()), tmp);
        copy(stdex::submdspan(tmp, std::tuple{0, output_size()}), out);
    }

    std::size_t _signal_size;
    std::size_t _patch_size;
    fft::rfft_plan<Float> _plan{fft::from_order, fft::next_order(output_size())};

    stdex::mdarray<Float, stdex::dextents<size_t, 1>> _tmp{_plan.size()};
    stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 1>> _signal_spectrum{_plan.size() / 2 + 1};
    stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 1>> _patch_spectrum{_plan.size() / 2 + 1};
};

template<in_vector Signal, in_vector Patch>
    requires(std::floating_point<value_type_t<Signal>> and std::floating_point<value_type_t<Patch>>)
auto fft_convolve(Signal signal, Patch patch) -> stdex::mdarray<value_type_t<Signal>, stdex::dextents<size_t, 1>>
{
    using Float = value_type_t<Signal>;

    if (signal.extent(0) == 0 or patch.extent(0) == 0) {
        return stdex::mdarray<Float, stdex::dextents<size_t, 1>>{};
    }

    auto convolver = fft_convolver<Float>{signal.extent(0), patch.extent(0)};
    auto output    = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{convolver.output_size()};
    convolver(signal, patch, output.to_mdspan());
    return output;
}

}  // namespace neo::convolution
