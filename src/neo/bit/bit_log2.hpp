// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <concepts>

namespace neo {

template<std::unsigned_integral UInt>
[[nodiscard]] constexpr auto bit_log2(UInt x) noexcept -> UInt
{
    auto result = UInt{0};
    for (; x > UInt(1); x >>= UInt(1)) {
        ++result;
    }
    return result;
}

}  // namespace neo
