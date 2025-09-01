// SPDX-License-Identifier: MIT

#include <neo/fft.hpp>

#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

template<typename Plan>
auto r2c(benchmark::State& state) -> void
{
    using Complex = typename Plan::complex_type;
    using Float   = typename Plan::real_type;

    auto const len   = static_cast<std::size_t>(state.range(0));
    auto const order = neo::fft::next_order(len);
    auto const noise = neo::generate_noise_signal<Float>(len, std::random_device{}());

    auto plan   = Plan{neo::fft::from_order, order};
    auto input  = noise;
    auto output = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{plan.size() / 2 + 1};

    for (auto _ : state) {
        state.PauseTiming();
        neo::copy(noise.to_mdspan(), input.to_mdspan());
        state.ResumeTiming();

        neo::fft::rfft(plan, input.to_mdspan(), output.to_mdspan());
        neo::fft::irfft(plan, output.to_mdspan(), input.to_mdspan());

        benchmark::DoNotOptimize(input.data());
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    auto const flop        = 5zu * plan.size() * size_t(plan.order()) * static_cast<size_t>(state.iterations());
    state.counters["flop"] = benchmark::Counter(static_cast<double>(flop), benchmark::Counter::kIsRate);
}

}  // namespace

#if defined(NEO_HAS_INTEL_IPP)
BENCHMARK(r2c<neo::fft::intel_ipp_rfft_plan<float>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);
BENCHMARK(r2c<neo::fft::intel_ipp_rfft_plan<double>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);
#endif

BENCHMARK(r2c<neo::fft::fallback_rfft_plan_v1<float>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);
BENCHMARK(r2c<neo::fft::fallback_rfft_plan_v1<double>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

BENCHMARK(r2c<neo::fft::fallback_rfft_plan_v2<float>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);
BENCHMARK(r2c<neo::fft::fallback_rfft_plan_v2<double>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

BENCHMARK(r2c<neo::fft::rfft_plan<float>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);
BENCHMARK(r2c<neo::fft::rfft_plan<double>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

BENCHMARK_MAIN();
