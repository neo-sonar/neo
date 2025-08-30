// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/bit/bit_ceil.hpp>
#include <neo/bit/bit_log2.hpp>
#include <neo/math/ipow.hpp>

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace neo::fft {

/// \ingroup neo-fft
struct from_order_tag
{
    explicit from_order_tag() = default;
};

/// \ingroup neo-fft
/// \relates from_order_tag
inline constexpr auto from_order = from_order_tag{};

/// \ingroup neo-fft
template<std::integral Int>
[[nodiscard]] constexpr auto next_order(Int size) -> Int
{
    assert(size > 0);
    auto const usize = static_cast<std::make_unsigned_t<Int>>(size);
    return static_cast<Int>(bit_log2(bit_ceil(usize)));
}

}  // namespace neo::fft
