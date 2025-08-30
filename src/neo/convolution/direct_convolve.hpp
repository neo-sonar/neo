// SPDX-License-Identifier: MIT

#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/convolution/mode.hpp>

#include <cassert>
#include <utility>

namespace neo::convolution {

/// \ingroup neo-convolution
template<in_vector Signal, in_vector Patch, out_vector Output>
    requires(std::same_as<typename Signal::value_type, typename Patch::value_type>)
auto direct_convolve(Signal signal, Patch patch, Output output) noexcept -> void
{
    auto const n  = signal.extent(0);
    auto const l  = patch.extent(0);
    auto const mm = output_size<mode::full>(n, l);
    assert(std::cmp_equal(output.extent(0), mm));

    if (n >= l) {
        auto i = size_t{0};
        for (auto k = size_t{0}; k < l; k++) {
            output[k] = 0.0;
            for (auto m = size_t{0}; m <= k; m++) {
                output[k] += signal[m] * patch[k - m];
            }
        }
        for (auto k = l; k < mm; k++) {
            output[k] = 0.0;
            i++;
            auto const t1   = l + i;
            auto const tmin = std::min(t1, n);
            for (auto m = i; m < tmin; m++) {
                output[k] += signal[m] * patch[k - m];
            }
        }
        return;
    }

    auto i = size_t{0};
    for (auto k = size_t{0}; k < n; k++) {
        output[k] = 0.0;
        for (auto m = size_t{0}; m <= k; m++) {
            output[k] += patch[m] * signal[k - m];
        }
    }
    for (auto k = n; k < mm; k++) {
        output[k] = 0.0;
        i++;
        auto const t1   = n + i;
        auto const tmin = std::min(t1, l);
        for (auto m = i; m < tmin; m++) {
            output[k] += patch[m] * signal[k - m];
        }
    }
}

/// \ingroup neo-convolution
template<in_vector Signal, in_vector Patch>
    requires(std::same_as<value_type_t<Signal>, value_type_t<Patch>>)
auto direct_convolve(Signal signal, Patch patch) -> stdex::mdarray<value_type_t<Signal>, stdex::dextents<size_t, 1>>
{
    using Float = value_type_t<Signal>;

    auto size   = output_size<mode::full>(signal.extent(0), patch.extent(0));
    auto output = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{size};
    direct_convolve(signal, patch, output.to_mdspan());
    return output;
}

}  // namespace neo::convolution
