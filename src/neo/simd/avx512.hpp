// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <immintrin.h>

namespace neo {

struct alignas(64) float32x16
{
    using value_type    = float;
    using register_type = __m512;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    float32x16() = default;

    float32x16(register_type val) noexcept : _val{val} {}

    [[nodiscard]] explicit operator register_type() const noexcept { return _val; }

    [[nodiscard]] static auto broadcast(float val) noexcept -> float32x16 { return _mm512_set1_ps(val); }

    [[nodiscard]] static auto load_unaligned(float const* input) noexcept -> float32x16
    {
        return _mm512_loadu_ps(input);
    }

    auto store_unaligned(float* output) const noexcept -> void { return _mm512_storeu_ps(output, _val); }

    NEO_ALWAYS_INLINE friend auto operator+(float32x16 lhs, float32x16 rhs) noexcept -> float32x16
    {
        return _mm512_add_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(float32x16 lhs, float32x16 rhs) noexcept -> float32x16
    {
        return _mm512_sub_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(float32x16 lhs, float32x16 rhs) noexcept -> float32x16
    {
        return _mm512_mul_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _val;
};

struct alignas(64) float64x8
{
    using value_type    = double;
    using register_type = __m512d;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    float64x8() = default;

    float64x8(register_type val) noexcept : _val{val} {}

    [[nodiscard]] explicit operator register_type() const noexcept { return _val; }

    [[nodiscard]] static auto broadcast(double val) noexcept -> float64x8 { return _mm512_set1_pd(val); }

    [[nodiscard]] static auto load_unaligned(double const* input) noexcept -> float64x8
    {
        return _mm512_loadu_pd(input);
    }

    auto store_unaligned(double* output) const noexcept -> void { return _mm512_storeu_pd(output, _val); }

    NEO_ALWAYS_INLINE friend auto operator+(float64x8 lhs, float64x8 rhs) noexcept -> float64x8
    {
        return _mm512_add_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(float64x8 lhs, float64x8 rhs) noexcept -> float64x8
    {
        return _mm512_sub_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(float64x8 lhs, float64x8 rhs) noexcept -> float64x8
    {
        return _mm512_mul_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _val;
};

}  // namespace neo
