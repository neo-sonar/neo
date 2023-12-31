// SPDX-License-Identifier: MIT

#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/fft/stft.hpp>
#include <neo/math/windowing.hpp>

namespace neo {

template<in_matrix InMat>
[[nodiscard]] auto uniform_partition(InMat impulse_response, std::size_t block_size)
{
    using Float = typename InMat::value_type;

    return fft::stft(
        impulse_response,
        {
            .frame_length   = static_cast<int>(block_size),
            .transform_size = static_cast<int>(block_size * 2UL),
            .overlap_length = 0,
            .window         = rectangular_window<Float>{},
        }
    );
}

}  // namespace neo
