// SPDX-License-Identifier: MIT

#pragma once

#include <neo/container/mdspan.hpp>

#include <cmath>
#include <concepts>
#include <numbers>

namespace neo {

/// \ingroup neo-math
template<std::floating_point Float>
struct rectangular_window
{
    using real_type = Float;

    rectangular_window() noexcept = default;

    [[nodiscard]] auto operator()(std::integral auto /*index*/, std::integral auto /*size*/) const noexcept -> Float
    {
        return Float(1.0);
    }
};

/// \ingroup neo-math
template<std::floating_point Float>
struct hann_window
{
    using real_type = Float;

    hann_window() noexcept = default;

    [[nodiscard]] auto operator()(std::integral auto index, std::integral auto size) const noexcept -> Float
    {
        auto const n      = static_cast<Float>(size - 1);
        auto const two_pi = static_cast<Float>(std::numbers::pi) * Float(2);
        return Float(0.5) * (Float(1) - std::cos(two_pi * static_cast<Float>(index) / n));
    }
};

/// \ingroup neo-math
template<std::floating_point Float>
struct hamming_window
{
    using real_type = Float;

    hamming_window() noexcept = default;

    [[nodiscard]] auto operator()(std::integral auto index, std::integral auto size) const noexcept -> Float
    {
        auto const n      = static_cast<Float>(size - 1);
        auto const two_pi = static_cast<Float>(std::numbers::pi) * Float(2);
        return Float(0.54) - Float(0.46) * std::cos(two_pi * static_cast<Float>(index) / n);
    }
};

/// \ingroup neo-math
auto fill_window(inout_vector auto vec, auto const& window)
{
    auto const size = static_cast<std::size_t>(vec.extent(0));
    for (auto i = std::size_t(0); i < size; ++i) {
        vec(i) = window(i, size);
    }
}

/// \ingroup neo-math
template<std::floating_point Float, typename Window = hann_window<Float>>
[[nodiscard]] auto generate_window(std::size_t length) -> stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>
{
    auto buffer = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{length};
    fill_window(buffer.to_mdspan(), Window{});
    return buffer;
}

}  // namespace neo
