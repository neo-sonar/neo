// SPDX-License-Identifier: MIT

#pragma once

#define NEO_STRINGIFY(x) #x

#if defined(_MSC_VER) and not defined(__clang__)
    #define NEO_COMPILER_MSVC
#elif defined(__GNUC__) and not defined(__clang__)
    #define NEO_COMPILER_GCC
#elif defined(__clang__)
    #define NEO_COMPILER_CLANG
#else
    #define NEO_COMPILER_UNKOWN
#endif

#if defined(__ARM_NEON__) or defined(__ARM_NEON)
    #define NEO_HAS_ISA_NEON
#endif

#if defined(__SSE2__) or defined(_M_AMD64) or defined(_M_X64)
    #define NEO_HAS_ISA_SSE2
#endif

#if defined(__SSE3__) or defined(__AVX__)
    #define NEO_HAS_ISA_SSE3
#endif

#if defined(__SSE4_1__) or defined(__AVX__)
    #define NEO_HAS_ISA_SSE41
#endif

#if defined(__AVX__)
    #define NEO_HAS_ISA_AVX
#endif

#if defined(__F16C__)
    #define NEO_HAS_ISA_F16C
#endif

#if defined(__AVX2__)
    #define NEO_HAS_ISA_AVX2
#endif

#if defined(__AVX512F__)
    #define NEO_HAS_ISA_AVX512F
#endif

#if defined(__AVX512BW__)
    #define NEO_HAS_ISA_AVX512BW
#endif

#if defined(__linux__) and not defined(__ANDROID__)
    #define NEO_PLATFORM_LINUX
#endif

#if defined(__ANDROID__)
    #define NEO_PLATFORM_ANDROID
#endif

#if defined(__FreeBSD__)
    #define NEO_PLATFORM_FREEBSD
#endif

#if defined(__OpenBSD__)
    #define NEO_PLATFORM_OPENBSD
#endif

#if defined(_WIN32)
    #define NEO_PLATFORM_WINDOWS
#endif

#if defined(__APPLE__)
    #if not defined(CF_EXCLUDE_CSTD_HEADERS)
        #define CF_EXCLUDE_CSTD_HEADERS
        #define NEO_CF_EXCLUDE_CSTD_HEADERS_WAS_NOT_DEFINED
    #endif

    #include <AvailabilityMacros.h>
    #include <TargetConditionals.h>

    #if defined(NEO_CF_EXCLUDE_CSTD_HEADERS_WAS_NOT_DEFINED)
        #undef CF_EXCLUDE_CSTD_HEADERS
    #endif

    #define NEO_PLATFORM_APPLE
    #if TARGET_OS_MAC == 1
        #define NEO_PLATFORM_MACOS
    #endif
#endif

#if defined(NEO_PLATFORM_APPLE)
    #define NEO_HAS_BUILTIN_FLOAT16
#elif defined(NEO_HAS_ISA_SSE41) and defined(NEO_COMPILER_CLANG)
    #define NEO_HAS_BUILTIN_FLOAT16
#endif

#if defined(__GNUC__) or defined(__clang__)
    #define NEO_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
    #define NEO_ALWAYS_INLINE __forceinline
#else
    #define NEO_ALWAYS_INLINE inline
#endif

#if defined(_MSC_VER)
    #define NEO_RESTRICT __restrict
#elif defined(__GNUC__) or defined(__clang__)
    #define NEO_RESTRICT __restrict__
#else
    #define NEO_RESTRICT
#endif
