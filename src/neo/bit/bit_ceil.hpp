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
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    if constexpr (sizeof(UInt) > 1) {
        x |= (x >> 8);
    }
    if constexpr (sizeof(UInt) > 2) {
        x |= (x >> 16);
    }
    if constexpr (sizeof(UInt) > 4) {
        x |= (x >> 32);
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
