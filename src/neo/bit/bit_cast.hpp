// SPDX-License-Identifier: MIT

#pragma once

#include <bit>
#include <cstring>
#include <type_traits>

namespace neo {

template<typename To, typename From>
    requires(sizeof(To) == sizeof(From) and std::is_trivially_copyable_v<From> and std::is_trivially_copyable_v<To>)
[[nodiscard]] constexpr auto bit_cast(From const& src) noexcept -> To
{
#if defined(__cpp_lib_bit_cast)
    return std::bit_cast<To>(src);
#elif __has_builtin(__builtin_bit_cast)
    return __builtin_bit_cast(To, src);
#else
    static_assert(std::is_trivially_constructible_v<To>);

    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
#endif
}

}  // namespace neo
