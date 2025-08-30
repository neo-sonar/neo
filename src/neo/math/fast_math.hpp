// SPDX-License-Identifier: MIT

#pragma once

#include <bit>
#include <cstdint>

namespace neo {

/// \ingroup neo-math
[[nodiscard]] constexpr auto fast_log2(float x) noexcept -> float
{
    auto const vx = std::bit_cast<std::uint32_t>(x);
    auto const mx = std::bit_cast<float>((vx & 0x007FFFFF) | 0x3f000000);
    auto const y  = static_cast<float>(vx) * 1.1920928955078125e-7F;
    return y - 124.22551499F - 1.498030302F * mx - 1.72587999F / (0.3520887068F + mx);
}

/// \ingroup neo-math
[[nodiscard]] constexpr auto fast_log10(float x) noexcept -> float
{
    constexpr auto scale = 0.30102999566F;  // 1 / log2(10);
    return fast_log2(x) * scale;
}

}  // namespace neo
