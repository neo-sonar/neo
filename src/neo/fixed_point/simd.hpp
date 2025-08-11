// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/fixed_point/fixed_point.hpp>

#if defined(NEO_HAS_ISA_SSE2)
    #include <neo/simd/sse2.hpp>
#endif

#if defined(NEO_HAS_ISA_AVX2)
    #include <neo/simd/avx.hpp>
#endif

#if defined(NEO_HAS_ISA_NEON)
    #include <neo/simd/neon.hpp>
#endif

#include <cassert>

namespace neo {

namespace detail {

#if defined(NEO_HAS_ISA_NEON)
inline constexpr auto const add_kernel_s8  = [](int8x16_t l, int8x16_t r) { return vqaddq_s8(l, r); };
inline constexpr auto const add_kernel_s16 = [](int16x8_t l, int16x8_t r) { return vqaddq_s16(l, r); };
inline constexpr auto const sub_kernel_s8  = [](int8x16_t l, int8x16_t r) { return vqsubq_s8(l, r); };
inline constexpr auto const sub_kernel_s16 = [](int16x8_t l, int16x8_t r) { return vqsubq_s16(l, r); };
inline constexpr auto const mul_kernel_s16 = [](int16x8_t l, int16x8_t r) { return vqdmulhq_s16(l, r); };
#elif defined(NEO_HAS_ISA_SSE2)
inline constexpr auto const add_kernel_s8  = [](__m128i l, __m128i r) { return _mm_adds_epi8(l, r); };
inline constexpr auto const add_kernel_s16 = [](__m128i l, __m128i r) { return _mm_adds_epi16(l, r); };
inline constexpr auto const sub_kernel_s8  = [](__m128i l, __m128i r) { return _mm_subs_epi8(l, r); };
inline constexpr auto const sub_kernel_s16 = [](__m128i l, __m128i r) { return _mm_subs_epi16(l, r); };
#endif

#if defined(NEO_HAS_ISA_SSE41)

template<int FractionalBits>
inline constexpr auto const mul_kernel_s8 = [](__m128i lhs, __m128i rhs) {
    auto const low_left    = _mm_cvtepi8_epi16(lhs);
    auto const low_right   = _mm_cvtepi8_epi16(rhs);
    auto const low_product = _mm_mullo_epi16(low_left, low_right);
    auto const low_shifted = _mm_srli_epi16(low_product, FractionalBits);

    auto const high_left    = _mm_cvtepi8_epi16(_mm_srli_si128(lhs, 8));
    auto const high_right   = _mm_cvtepi8_epi16(_mm_srli_si128(rhs, 8));
    auto const high_product = _mm_mullo_epi16(high_left, high_right);
    auto const high_shifted = _mm_srli_epi16(high_product, FractionalBits);

    return _mm_packs_epi16(low_shifted, high_shifted);
};

template<int FractionalBits>
inline constexpr auto const mul_kernel_s16 = [](__m128i lhs, __m128i rhs) {
    if constexpr (FractionalBits == 15) {
        return _mm_mulhrs_epi16(lhs, rhs);
    } else {
        auto const low_left    = _mm_cvtepi16_epi32(lhs);
        auto const low_right   = _mm_cvtepi16_epi32(rhs);
        auto const low_product = _mm_mullo_epi32(low_left, low_right);
        auto const low_shifted = _mm_srli_epi32(low_product, FractionalBits);

        auto const high_left    = _mm_cvtepi16_epi32(_mm_srli_si128(lhs, 8));
        auto const high_right   = _mm_cvtepi16_epi32(_mm_srli_si128(rhs, 8));
        auto const high_product = _mm_mullo_epi32(high_left, high_right);
        auto const high_shifted = _mm_srli_epi32(high_product, FractionalBits);

        return _mm_packs_epi32(low_shifted, high_shifted);
    }
};
#endif

#if defined(NEO_HAS_ISA_AVX2)

template<int FractionalBits>
inline constexpr auto const mul_kernel_avx2_s16 = [](__m256i lhs, __m256i rhs) {
    if constexpr (FractionalBits == 15) {
        return _mm256_mulhrs_epi16(lhs, rhs);
    } else {
        assert(false);
    }
};

#endif

#if defined(NEO_HAS_ISA_AVX512BW)

template<int FractionalBits>
inline constexpr auto const mul_kernel_avx512_s16 = [](__m512i lhs, __m512i rhs) {
    if constexpr (FractionalBits == 15) {
        return _mm512_mulhrs_epi16(lhs, rhs);
    } else {
        assert(false);
    }
};

#endif

}  // namespace detail

#if defined(NEO_HAS_ISA_NEON)

struct alignas(16) q7x16
{
    using value_type    = neo::q7;
    using register_type = int8x16_t;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q7x16() = default;

    NEO_ALWAYS_INLINE q7x16(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) noexcept -> q7x16
    {
        auto const v = val.value();
        return vld1q_dup_s8(&v);
    }

    [[nodiscard]] static auto load_unaligned(value_type const* input) noexcept -> q7x16
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        return vld1q_s8(integer);
    }

    auto store_unaligned(value_type* output) const noexcept -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        return vst1q_s8(integer, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q7x16 lhs, q7x16 rhs) noexcept -> q7x16
    {
        return vqaddq_s8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q7x16 lhs, q7x16 rhs) noexcept -> q7x16
    {
        return vqsubq_s8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

struct alignas(16) q15x8
{
    using value_type    = neo::q15;
    using register_type = int16x8_t;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q15x8() = default;

    NEO_ALWAYS_INLINE q15x8(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) noexcept -> q15x8
    {
        auto const v = val.value();
        return vld1q_dup_s16(&v);
    }

    [[nodiscard]] static auto load_unaligned(value_type const* input) noexcept -> q15x8
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        return vld1q_s16(integer);
    }

    auto store_unaligned(value_type* output) const noexcept -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        return vst1q_s16(integer, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q15x8 lhs, q15x8 rhs) noexcept -> q15x8
    {
        return vqaddq_s16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q15x8 lhs, q15x8 rhs) noexcept -> q15x8
    {
        return vqsubq_s16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(q15x8 lhs, q15x8 rhs) noexcept -> q15x8
    {
        return vqdmulhq_s16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

#endif

#if defined(NEO_HAS_ISA_SSE2)

struct alignas(16) q7x16
{
    using value_type    = neo::q7;
    using register_type = __m128i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q7x16() = default;

    NEO_ALWAYS_INLINE q7x16(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) noexcept -> q7x16 { return _mm_set1_epi8(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) noexcept -> q7x16
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm_loadu_si128(ptr);
    }

    auto store_unaligned(value_type* output) const noexcept -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        _mm_storeu_si128(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q7x16 lhs, q7x16 rhs) noexcept -> q7x16
    {
        return _mm_adds_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q7x16 lhs, q7x16 rhs) noexcept -> q7x16
    {
        return _mm_subs_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    #if defined(NEO_HAS_ISA_SSE41)
    NEO_ALWAYS_INLINE friend auto operator*(q7x16 lhs, q7x16 rhs) noexcept -> q7x16
    {
        return detail::mul_kernel_s8<value_type::fractional_bits>(
            static_cast<register_type>(lhs),
            static_cast<register_type>(rhs)
        );
    }
    #endif

private:
    register_type _reg;
};

struct alignas(16) q15x8
{
    using value_type    = neo::q15;
    using register_type = __m128i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q15x8() = default;

    NEO_ALWAYS_INLINE q15x8(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) noexcept -> q15x8 { return _mm_set1_epi16(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) noexcept -> q15x8
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm_loadu_si128(ptr);
    }

    auto store_unaligned(value_type* output) const noexcept -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        _mm_storeu_si128(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q15x8 lhs, q15x8 rhs) noexcept -> q15x8
    {
        return _mm_adds_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q15x8 lhs, q15x8 rhs) noexcept -> q15x8
    {
        return _mm_subs_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    #if defined(NEO_HAS_ISA_SSE41)
    NEO_ALWAYS_INLINE friend auto operator*(q15x8 lhs, q15x8 rhs) noexcept -> q15x8
    {
        return detail::mul_kernel_s16<value_type::fractional_bits>(
            static_cast<register_type>(lhs),
            static_cast<register_type>(rhs)
        );
    }
    #endif

private:
    register_type _reg;
};

#endif

#if defined(NEO_HAS_ISA_AVX2)

struct alignas(32) q7x32
{
    using value_type    = neo::q7;
    using register_type = __m256i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q7x32() = default;

    NEO_ALWAYS_INLINE q7x32(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) noexcept -> q7x32 { return _mm256_set1_epi8(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) noexcept -> q7x32
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm256_loadu_si256(ptr);
    }

    auto store_unaligned(value_type* output) const noexcept -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        _mm256_storeu_si256(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q7x32 lhs, q7x32 rhs) noexcept -> q7x32
    {
        return _mm256_adds_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q7x32 lhs, q7x32 rhs) noexcept -> q7x32
    {
        return _mm256_subs_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

struct alignas(32) q15x16
{
    using value_type    = neo::q15;
    using register_type = __m256i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q15x16() = default;

    NEO_ALWAYS_INLINE q15x16(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) noexcept -> q15x16 { return _mm256_set1_epi16(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) noexcept -> q15x16
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm256_loadu_si256(ptr);
    }

    auto store_unaligned(value_type* output) const noexcept -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        _mm256_storeu_si256(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q15x16 lhs, q15x16 rhs) noexcept -> q15x16
    {
        return _mm256_adds_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q15x16 lhs, q15x16 rhs) noexcept -> q15x16
    {
        return _mm256_subs_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(q15x16 lhs, q15x16 rhs) noexcept -> q15x16
    {
        return detail::mul_kernel_avx2_s16<value_type::fractional_bits>(
            static_cast<register_type>(lhs),
            static_cast<register_type>(rhs)
        );
    }

private:
    register_type _reg;
};

#endif

#if defined(NEO_HAS_ISA_AVX512BW)

struct alignas(64) q7x64
{
    using value_type    = neo::q7;
    using register_type = __m512i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q7x64() = default;

    NEO_ALWAYS_INLINE q7x64(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) noexcept -> q7x64 { return _mm512_set1_epi8(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) noexcept -> q7x64
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm512_loadu_si512(ptr);
    }

    auto store_unaligned(value_type* output) const noexcept -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        return _mm512_storeu_si512(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q7x64 lhs, q7x64 rhs) noexcept -> q7x64
    {
        return _mm512_adds_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q7x64 lhs, q7x64 rhs) noexcept -> q7x64
    {
        return _mm512_subs_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

struct alignas(64) q15x32
{
    using value_type    = neo::q15;
    using register_type = __m512i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q15x32() = default;

    NEO_ALWAYS_INLINE q15x32(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) noexcept -> q15x32 { return _mm512_set1_epi16(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) noexcept -> q15x32
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm512_loadu_si512(ptr);
    }

    auto store_unaligned(value_type* output) const noexcept -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        return _mm512_storeu_si512(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q15x32 lhs, q15x32 rhs) noexcept -> q15x32
    {
        return _mm512_adds_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q15x32 lhs, q15x32 rhs) noexcept -> q15x32
    {
        return _mm512_subs_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(q15x32 lhs, q15x32 rhs) noexcept -> q15x32
    {
        return detail::mul_kernel_avx512_s16<value_type::fractional_bits>(
            static_cast<register_type>(lhs),
            static_cast<register_type>(rhs)
        );
    }

private:
    register_type _reg;
};

#endif

}  // namespace neo
