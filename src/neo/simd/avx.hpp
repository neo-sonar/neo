// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <neo/config.hpp>

#include <immintrin.h>

// #define __STDC_WANT_IEC_60559_TYPES_EXT__ 1
#include <array>
#include <cfloat>

namespace neo {

#if defined(NEO_HAS_BUILTIN_FLOAT16) and defined(NEO_HAS_ISA_F16C)
struct alignas(16) float16x8
{
    using value_type    = _Float16;
    using register_type = __m128i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    float16x8() = default;

    float16x8(register_type val) noexcept : _register{val} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _register; }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto broadcast(value_type val) noexcept -> float16x8
    {
        auto values = std::array<_Float16, size>{};
        std::ranges::fill(values, val);
        return load_unaligned(values.data());
    }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto load_unaligned(value_type const* input) noexcept -> float16x8
    {
        return _mm_loadu_si128(reinterpret_cast<register_type const*>(input));
    }

    NEO_ALWAYS_INLINE auto store_unaligned(value_type* output) const noexcept -> void
    {
        _mm_storeu_si128(reinterpret_cast<register_type*>(output), _register);
    }

    NEO_ALWAYS_INLINE friend auto operator+(float16x8 lhs, float16x8 rhs) noexcept -> float16x8
    {
        return binary_op(lhs, rhs, [](auto l, auto r) { return _mm256_add_ps(l, r); });
    }

    NEO_ALWAYS_INLINE friend auto operator-(float16x8 lhs, float16x8 rhs) noexcept -> float16x8
    {
        return binary_op(lhs, rhs, [](auto l, auto r) { return _mm256_sub_ps(l, r); });
    }

    NEO_ALWAYS_INLINE friend auto operator*(float16x8 lhs, float16x8 rhs) noexcept -> float16x8
    {
        return binary_op(lhs, rhs, [](auto l, auto r) { return _mm256_mul_ps(l, r); });
    }

private:
    NEO_ALWAYS_INLINE friend auto binary_op(float16x8 lhs, float16x8 rhs, auto op) noexcept -> float16x8
    {
        auto const left    = _mm256_cvtph_ps(static_cast<register_type>(lhs));
        auto const right   = _mm256_cvtph_ps(static_cast<register_type>(rhs));
        auto const product = op(left, right);
        return _mm256_cvtps_ph(product, _MM_FROUND_CUR_DIRECTION);
    }

    register_type _register;
};

struct alignas(32) float16x16
{
    using value_type    = _Float16;
    using register_type = __m256i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    float16x16() = default;

    float16x16(register_type val) noexcept : _register{val} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _register; }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto broadcast(value_type val) noexcept -> float16x16
    {
        auto values = std::array<_Float16, size>{};
        std::ranges::fill(values, val);
        return load_unaligned(values.data());
    }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto load_unaligned(value_type const* input) noexcept -> float16x16
    {
        return _mm256_loadu_si256(reinterpret_cast<register_type const*>(input));
    }

    NEO_ALWAYS_INLINE auto store_unaligned(value_type* output) const noexcept -> void
    {
        _mm256_storeu_si256(reinterpret_cast<register_type*>(output), _register);
    }

    NEO_ALWAYS_INLINE friend auto operator+(float16x16 lhs, float16x16 rhs) noexcept -> float16x16
    {
        return binary_op(lhs, rhs, [](auto l, auto r) { return _mm256_add_ps(l, r); });
    }

    NEO_ALWAYS_INLINE friend auto operator-(float16x16 lhs, float16x16 rhs) noexcept -> float16x16
    {
        return binary_op(lhs, rhs, [](auto l, auto r) { return _mm256_sub_ps(l, r); });
    }

    NEO_ALWAYS_INLINE friend auto operator*(float16x16 lhs, float16x16 rhs) noexcept -> float16x16
    {
        return binary_op(lhs, rhs, [](auto l, auto r) { return _mm256_mul_ps(l, r); });
    }

private:
    NEO_ALWAYS_INLINE friend auto binary_op(float16x16 lhs, float16x16 rhs, auto op) noexcept -> float16x16
    {
        auto const low_left   = _mm256_extracti128_si256(static_cast<register_type>(lhs), 0);
        auto const high_left  = _mm256_extracti128_si256(static_cast<register_type>(lhs), 1);
        auto const low_right  = _mm256_extracti128_si256(static_cast<register_type>(rhs), 0);
        auto const high_right = _mm256_extracti128_si256(static_cast<register_type>(rhs), 1);

        auto const low_left_f16   = _mm256_cvtph_ps(low_left);
        auto const high_left_f16  = _mm256_cvtph_ps(high_left);
        auto const low_right_f16  = _mm256_cvtph_ps(low_right);
        auto const high_right_f16 = _mm256_cvtph_ps(high_right);

        auto const low_product  = _mm256_cvtps_ph(op(low_left_f16, low_right_f16), _MM_FROUND_CUR_DIRECTION);
        auto const high_product = _mm256_cvtps_ph(op(high_left_f16, high_right_f16), _MM_FROUND_CUR_DIRECTION);

        return _mm256_insertf128_si256(_mm256_castsi128_si256(low_product), high_product, 1);
    }

    register_type _register;
};

#endif

struct alignas(32) float32x8
{
    using value_type    = float;
    using register_type = __m256;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    float32x8() = default;

    NEO_ALWAYS_INLINE float32x8(register_type val) noexcept : _val{val} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _val; }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto broadcast(float val) noexcept -> float32x8
    {
        return _mm256_set1_ps(val);
    }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto load_unaligned(float const* input) noexcept -> float32x8
    {
        return _mm256_loadu_ps(input);
    }

    NEO_ALWAYS_INLINE auto store_unaligned(float* output) const noexcept -> void { _mm256_storeu_ps(output, _val); }

    NEO_ALWAYS_INLINE friend auto operator+(float32x8 lhs, float32x8 rhs) noexcept -> float32x8
    {
        return _mm256_add_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(float32x8 lhs, float32x8 rhs) noexcept -> float32x8
    {
        return _mm256_sub_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(float32x8 lhs, float32x8 rhs) noexcept -> float32x8
    {
        return _mm256_mul_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _val;
};

struct alignas(32) float64x4
{
    using value_type    = double;
    using register_type = __m256d;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    float64x4() = default;

    NEO_ALWAYS_INLINE float64x4(register_type val) noexcept : _val{val} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _val; }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto broadcast(double val) noexcept -> float64x4
    {
        return _mm256_set1_pd(val);
    }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto load_unaligned(double const* input) noexcept -> float64x4
    {
        return _mm256_loadu_pd(input);
    }

    NEO_ALWAYS_INLINE auto store_unaligned(double* output) const noexcept -> void { _mm256_storeu_pd(output, _val); }

    NEO_ALWAYS_INLINE friend auto operator+(float64x4 lhs, float64x4 rhs) noexcept -> float64x4
    {
        return _mm256_add_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(float64x4 lhs, float64x4 rhs) noexcept -> float64x4
    {
        return _mm256_sub_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(float64x4 lhs, float64x4 rhs) noexcept -> float64x4
    {
        return _mm256_mul_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _val;
};

}  // namespace neo
