// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/type_traits/always_false.hpp>

namespace neo::convolution {

enum struct mode
{
    full,
    valid,
    same,
};

template<mode Mode, std::integral Int>
[[nodiscard]] auto output_size(Int signal, Int patch) -> Int
{
    if constexpr (Mode == mode::full) {
        return static_cast<Int>(signal + patch - Int(1));
    } else {
        static_assert(always_false<Int>);
    }
}

}  // namespace neo::convolution
