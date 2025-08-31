#pragma once

#include <neo/config/xsimd.hpp>

#include <algorithm>
#include <array>
#include <functional>

namespace neo {

template<typename Float, std::size_t Size>
struct multi_batch
{
    using real_type  = Float;
    using batch_type = xsimd::batch<Float>;

    static constexpr auto size = Size * xsimd::batch<Float>::size;

    multi_batch() = default;

    multi_batch(batch_type batch) { std::ranges::fill(batches, batch); }

    auto operator=(batch_type batch) -> multi_batch&
    {
        (*this) = multi_batch{batch};
        return *this;
    }

    friend auto operator+(multi_batch const& lhs, multi_batch const& rhs) -> multi_batch
    {
        auto out = multi_batch{};
        std::ranges::transform(lhs.batches, rhs.batches, out.batches.begin(), std::plus());
        return out;
    }

    friend auto operator-(multi_batch const& lhs, multi_batch const& rhs) -> multi_batch
    {
        auto out = multi_batch{};
        std::ranges::transform(lhs.batches, rhs.batches, out.batches.begin(), std::minus());
        return out;
    }

    friend auto operator*(multi_batch const& lhs, multi_batch const& rhs) -> multi_batch
    {
        auto out = multi_batch{};
        std::ranges::transform(lhs.batches, rhs.batches, out.batches.begin(), std::multiplies());
        return out;
    }

    friend auto fms(multi_batch const& x, multi_batch const& y, multi_batch const& z) -> multi_batch
    {
        auto out = multi_batch{};
        for (auto i{0zu}; i < Size; ++i) {
            out.batches[i] = xsimd::fms(x.batches[i], y.batches[i], z.batches[i]);
        }
        return out;
    }

    friend auto fma(multi_batch const& x, multi_batch const& y, multi_batch const& z) -> multi_batch
    {
        auto out = multi_batch{};
        for (auto i{0zu}; i < Size; ++i) {
            out.batches[i] = xsimd::fma(x.batches[i], y.batches[i], z.batches[i]);
        }
        return out;
    }

    std::array<batch_type, Size> batches;
};

template<typename T>
inline constexpr bool is_multi_batch = false;

template<typename Float, std::size_t Size>
inline constexpr bool is_multi_batch<multi_batch<Float, Size>> = true;

}  // namespace neo
