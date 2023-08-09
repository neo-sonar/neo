#pragma once

#include <cmath>
#include <concepts>

namespace neo::fft {

template<std::floating_point Float>
[[nodiscard]] auto to_decibels(Float gain, Float infinity = Float(-144)) -> Float
{
    if (gain <= Float(0)) { return infinity; }
    return std::max(infinity, Float(20) * std::log10(gain));
}

}  // namespace neo::fft