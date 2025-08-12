// SPDX-License-Identifier: MIT

#pragma once

#include <bit>
#include <concepts>

namespace neo {

namespace detail {

template<std::unsigned_integral UInt>
[[nodiscard]] constexpr auto bit_ceil_fallback(UInt x) noexcept -> UInt
{
    --x;
    x = static_cast<UInt>(x | static_cast<UInt>((x >> UInt(1))));
    x = static_cast<UInt>(x | static_cast<UInt>((x >> UInt(2))));
    x = static_cast<UInt>(x | static_cast<UInt>((x >> UInt(4))));
    if constexpr (sizeof(UInt) > 1) {
        x = static_cast<UInt>(x | static_cast<UInt>((x >> UInt(8))));
    }
    if constexpr (sizeof(UInt) > 2) {
        x = static_cast<UInt>(x | static_cast<UInt>((x >> UInt(16))));
    }
    if constexpr (sizeof(UInt) > 4) {
        x = static_cast<UInt>(x | static_cast<UInt>((x >> UInt(32))));
    }
    return x + 1;
}

}  // namespace detail

template<std::unsigned_integral UInt>
[[nodiscard]] constexpr auto bit_ceil(UInt x) noexcept -> UInt
{
#if defined(__cpp_lib_int_pow2)
    return std::bit_ceil(x);
#else
    return detail::bit_ceil_fallback(x);
#endif
}

}  // namespace neo
