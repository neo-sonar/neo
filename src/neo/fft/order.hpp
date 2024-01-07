// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/bit/bit_ceil.hpp>
#include <neo/bit/bit_log2.hpp>
#include <neo/math/ipow.hpp>

#include <cstddef>
#include <type_traits>

namespace neo::fft {

enum struct order : std::size_t
{
};

[[nodiscard]] constexpr auto size(order o) noexcept -> std::underlying_type_t<order>
{
    using Int = std::underlying_type_t<order>;
    return ipow<Int(2)>(static_cast<Int>(o));
}

template<std::integral Int>
[[nodiscard]] constexpr auto next_order(Int size) noexcept -> order
{
    auto const usize = static_cast<std::make_unsigned_t<Int>>(size);
    return static_cast<order>(bit_log2(bit_ceil(usize)));
}

}  // namespace neo::fft
