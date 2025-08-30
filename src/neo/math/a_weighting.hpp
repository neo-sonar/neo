// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cmath>
#include <concepts>

namespace neo {

/// \pre \p frequency must be greater 0.0
/// \ingroup neo-math
template<std::floating_point Float>
[[nodiscard]] auto a_weighting(Float frequency) noexcept -> Float
{
    static constexpr auto const constants = []() -> std::array<Float, 4> {
        auto c = std::array<Float, 4>{Float(12194.217), Float(20.598997), Float(107.65265), Float(737.86223)};
        for (auto& v : c) {
            v = v * v;
        }
        return c;
    }();

    // clang-format off
    auto const f_sq = frequency * frequency;
    return Float(2) + Float(20) * (
          std::log10(constants[0])
        + Float(2) * std::log10(f_sq)
        - std::log10(f_sq + constants[0])
        - std::log10(f_sq + constants[1])
        - Float(0.5) * std::log10(f_sq + constants[2])
        - Float(0.5) * std::log10(f_sq + constants[3])
    );
    // clang-format on
}

}  // namespace neo
