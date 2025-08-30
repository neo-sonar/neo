// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>

namespace neo {

/// \ingroup neo-math
template<std::integral T>
[[nodiscard]] constexpr auto ipow(T base, T exponent) -> T
{
    T result = 1;
    for (T i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}

/// \ingroup neo-math
template<auto Base>
    requires std::integral<decltype(Base)>
[[nodiscard]] constexpr auto ipow(decltype(Base) exponent) -> decltype(Base)
{
    using Int  = decltype(Base);
    using UInt = std::make_unsigned_t<Int>;

    if constexpr (Base == Int(2)) {
        return static_cast<Int>(UInt(1) << UInt(exponent));
    } else if constexpr (Base == Int(4)) {
        return static_cast<Int>(UInt(1) << UInt(exponent * Int(2)));
    } else if constexpr (Base == Int(8)) {
        return static_cast<Int>(UInt(1) << UInt(exponent * Int(3)));
    } else if constexpr (Base == Int(16)) {
        return static_cast<Int>(UInt(1) << UInt(exponent * Int(4)));
    } else if constexpr (Base == Int(32)) {
        return static_cast<Int>(UInt(1) << UInt(exponent * Int(5)));
    } else {
        return ipow(Base, exponent);
    }
}

}  // namespace neo
