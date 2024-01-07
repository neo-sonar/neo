// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <concepts>

#if defined(NEO_HAS_XSIMD)
    #include <neo/config/xsimd.hpp>
#endif

namespace neo {

template<typename T>
inline constexpr auto const is_complex = false;

template<std::floating_point Float>
inline constexpr auto const is_complex<std::complex<Float>> = true;

#if defined(NEO_HAS_XSIMD)
template<std::floating_point Float>
inline constexpr auto const is_complex<xsimd::batch<std::complex<Float>>> = true;
#endif

template<typename T>
concept complex = is_complex<T>;

}  // namespace neo
