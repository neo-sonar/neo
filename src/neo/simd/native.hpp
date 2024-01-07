// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#if defined(NEO_HAS_ISA_NEON)
    #include <neo/simd/neon.hpp>
#endif

#if defined(NEO_HAS_ISA_SSE2)
    #include <neo/simd/sse2.hpp>
#endif

#if defined(NEO_HAS_ISA_AVX)
    #include <neo/simd/avx.hpp>
#endif

#if defined(NEO_HAS_ISA_AVX512F)
    #include <neo/simd/avx512.hpp>
#endif

namespace neo {

#if defined(NEO_HAS_ISA_AVX512F)
using float32x = float32x16;
using float64x = float64x8;
#elif defined(NEO_HAS_ISA_AVX)
using float32x = float32x8;
using float64x = float64x4;
#elif defined(NEO_HAS_ISA_SSE2)
using float32x = float32x4;
using float64x = float64x2;
#endif

}  // namespace neo
