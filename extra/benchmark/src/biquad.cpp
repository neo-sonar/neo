// SPDX-License-Identifier: MIT

#include "multi_batch.hpp"

#include <neo/fft.hpp>

#include <neo/config/xsimd.hpp>
#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <functional>
#include <ranges>

// #include <stdfloat>

namespace {

template<typename Float>
constexpr auto makeBypass() -> std::array<Float, 6>
{
    return {Float(1), Float(0), Float(0), Float(1), Float(0), Float(0)};
}

template<typename Float>
struct Biquad
{
    auto operator()(Float x) -> Float
    {
        using Index = Biquad<Float>::Index;

        auto const b0 = coefficients[Index::B0];
        auto const b1 = coefficients[Index::B1];
        auto const b2 = coefficients[Index::B2];
        auto const a1 = coefficients[Index::A1];
        auto const a2 = coefficients[Index::A2];

        auto const y = b0 * x + z[0];
        z[0]         = b1 * x - a1 * y + z[1];
        z[1]         = b2 * x - a2 * y;
        return y;
    }

private:
    enum Index
    {
        B0,
        B1,
        B2,
        A0,
        A1,
        A2,
        NumCoefficients,
    };

    std::array<Float, Index::NumCoefficients> coefficients{makeBypass<Float>()};
    std::array<Float, 2> z{};
};

template<typename Float>
auto biquad(benchmark::State& state) -> void
{
    auto const size = static_cast<std::size_t>(state.range(0));
    auto const sig  = [size] {
        if constexpr (xsimd::is_batch<Float>::value) {
            auto s = neo::generate_noise_signal<typename Float::value_type>(size, std::random_device{}());
            auto x = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{size};
            for (auto i{0zu}; i < size; ++i) {
                x(i) = xsimd::broadcast(s(i));
            }
            return x;
        } else if constexpr (neo::is_multi_batch<Float>) {
            auto s = neo::generate_noise_signal<typename Float::real_type>(size, std::random_device{}());
            auto x = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{size};
            for (auto i{0zu}; i < size; ++i) {
                x(i) = xsimd::broadcast(s(i));
            }
            return x;
        } else {
            return neo::generate_noise_signal<Float>(size, std::random_device{}());
        }
    }();

    auto filter = Biquad<Float>{};
    auto out    = std::vector<Float>(size);

    for (auto _ : state) {
        for (auto i{0zu}; i < size; ++i) {
            out[i] = filter(sig(i));
        }
        benchmark::DoNotOptimize(out[0]);
        benchmark::DoNotOptimize(out[size - 1]);
        benchmark::ClobberMemory();
    }

    auto const items = int64_t(state.iterations()) * int64_t(size);
    state.SetBytesProcessed(items * sizeof(Float));
    if constexpr (xsimd::is_batch<Float>::value or neo::is_multi_batch<Float>) {
        state.SetItemsProcessed(items * Float::size);
    } else {
        state.SetItemsProcessed(items);
    }
}

}  // namespace

static constexpr auto N = 2048 * 1024;

// #if defined(NEO_HAS_BUILTIN_FLOAT16)
// BENCHMARK(biquad<_Float16>)->RangeMultiplier(2)->Range(256, N);
// #endif

BENCHMARK(biquad<float>)->RangeMultiplier(2)->Range(256, N);
// BENCHMARK(biquad<double>)->RangeMultiplier(2)->Range(256, N);
// BENCHMARK(biquad<long double>)->RangeMultiplier(2)->Range(256, N);

BENCHMARK(biquad<xsimd::batch<float>>)->RangeMultiplier(2)->Range(256, N);
// BENCHMARK(biquad<xsimd::batch<double>>)->RangeMultiplier(2)->Range(256, N);

BENCHMARK(biquad<neo::multi_batch<float, 2>>)->RangeMultiplier(2)->Range(256, N);
BENCHMARK(biquad<neo::multi_batch<float, 4>>)->RangeMultiplier(2)->Range(256, N);
BENCHMARK(biquad<neo::multi_batch<float, 8>>)->RangeMultiplier(2)->Range(256, N);

// BENCHMARK(biquad<std::float32_t>)->RangeMultiplier(2)->Range(256, N);
// BENCHMARK(biquad<std::float64_t>)->RangeMultiplier(2)->Range(256, N);
// BENCHMARK(biquad<std::float128_t>)->RangeMultiplier(2)->Range(256, N);

BENCHMARK_MAIN();
