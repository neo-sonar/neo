// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>

namespace neo {

struct alignas(16) float32x4
{
    using value_type    = float;
    using register_type = __m128;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    float32x4() = default;

    NEO_ALWAYS_INLINE float32x4(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto broadcast(float reg) noexcept -> float32x4 { return _mm_set1_ps(reg); }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto load_unaligned(float const* input) noexcept -> float32x4
    {
        return _mm_loadu_ps(input);
    }

    NEO_ALWAYS_INLINE auto store_unaligned(float* output) const noexcept -> void { _mm_storeu_ps(output, _reg); }

    NEO_ALWAYS_INLINE friend auto operator+(float32x4 lhs, float32x4 rhs) noexcept -> float32x4
    {
        return _mm_add_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(float32x4 lhs, float32x4 rhs) noexcept -> float32x4
    {
        return _mm_sub_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(float32x4 lhs, float32x4 rhs) noexcept -> float32x4
    {
        return _mm_mul_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

struct alignas(16) float64x2
{
    using value_type    = double;
    using register_type = __m128d;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    float64x2() = default;

    NEO_ALWAYS_INLINE float64x2(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto broadcast(double reg) noexcept -> float64x2 { return _mm_set1_pd(reg); }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto load_unaligned(double const* input) noexcept -> float64x2
    {
        return _mm_loadu_pd(input);
    }

    NEO_ALWAYS_INLINE auto store_unaligned(double* output) const -> void { _mm_storeu_pd(output, _reg); }

    NEO_ALWAYS_INLINE friend auto operator+(float64x2 lhs, float64x2 rhs) noexcept -> float64x2
    {
        return _mm_add_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(float64x2 lhs, float64x2 rhs) noexcept -> float64x2
    {
        return _mm_sub_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(float64x2 lhs, float64x2 rhs) noexcept -> float64x2
    {
        return _mm_mul_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

namespace simd {

template<typename ScalarType>
inline constexpr auto apply_kernel = [](auto lhs, auto rhs, auto out, auto scalar_kernel, auto vector_kernel) {
    static constexpr auto value_size_bits = sizeof(ScalarType) * 8zu;
    static constexpr auto vector_size     = static_cast<ptrdiff_t>(128zu / value_size_bits);
    auto const remainder                  = static_cast<ptrdiff_t>(lhs.size()) % vector_size;

    for (auto i{0}; i < remainder; ++i) {
        out[static_cast<size_t>(i)] = scalar_kernel(lhs[static_cast<size_t>(i)], rhs[static_cast<size_t>(i)]);
    }

    for (auto i{remainder}; i < std::ssize(lhs); i += vector_size) {
        auto const left  = _mm_loadu_si128(reinterpret_cast<__m128i const*>(std::next(lhs.data(), i)));
        auto const right = _mm_loadu_si128(reinterpret_cast<__m128i const*>(std::next(rhs.data(), i)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(std::next(out.data(), i)), vector_kernel(left, right));
    }
};

}  // namespace simd

}  // namespace neo
