// SPDX-License-Identifier: MIT

#include <neo/algorithm.hpp>
#include <neo/fixed_point.hpp>

#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

template<typename ValueType>
auto multiply_add(benchmark::State& state) -> void
{
    auto const size = static_cast<std::size_t>(state.range(0));

    auto const x_buf = neo::generate_noise_signal<ValueType>(size, std::random_device{}());
    auto const y_buf = neo::generate_noise_signal<ValueType>(size, std::random_device{}());
    auto const z_buf = neo::generate_noise_signal<ValueType>(size, std::random_device{}());
    auto out_buf     = stdex::mdarray<ValueType, stdex::dextents<size_t, 1>>{size};

    auto const x   = x_buf.to_mdspan();
    auto const y   = y_buf.to_mdspan();
    auto const z   = z_buf.to_mdspan();
    auto const out = out_buf.to_mdspan();

    for (auto _ : state) {
        state.PauseTiming();
        neo::copy(z, out);
        state.ResumeTiming();

        neo::multiply_add(x, y, out, out);

        benchmark::DoNotOptimize(out[0]);
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * state.range(0);
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(ValueType));
}

template<typename Real>
auto split_multiply_add(benchmark::State& state) -> void
{
    auto const size = state.range(0);

    auto x_buf   = stdex::mdarray<Real, stdex::dextents<size_t, 2>>{2, size};
    auto y_buf   = stdex::mdarray<Real, stdex::dextents<size_t, 2>>{2, size};
    auto out_buf = stdex::mdarray<Real, stdex::dextents<size_t, 2>>{2, size};

    auto const x = neo::split_complex{
        stdex::submdspan(x_buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(x_buf.to_mdspan(), 1, stdex::full_extent),
    };
    auto const y = neo::split_complex{
        stdex::submdspan(y_buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(y_buf.to_mdspan(), 1, stdex::full_extent),
    };
    auto const out = neo::split_complex{
        stdex::submdspan(out_buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(out_buf.to_mdspan(), 1, stdex::full_extent),
    };

    auto const noise_x_real   = neo::generate_noise_signal<Real>(size, std::random_device{}());
    auto const noise_x_imag   = neo::generate_noise_signal<Real>(size, std::random_device{}());
    auto const noise_y_real   = neo::generate_noise_signal<Real>(size, std::random_device{}());
    auto const noise_y_imag   = neo::generate_noise_signal<Real>(size, std::random_device{}());
    auto const noise_out_real = neo::generate_noise_signal<Real>(size, std::random_device{}());
    auto const noise_out_imag = neo::generate_noise_signal<Real>(size, std::random_device{}());

    for (auto i{0U}; i < size; ++i) {
        x.real[i] = noise_x_real(i);
        x.imag[i] = noise_x_imag(i);
        y.real[i] = noise_y_real(i);
        y.imag[i] = noise_y_imag(i);
    }

    for (auto _ : state) {
        state.PauseTiming();
        for (auto i{0U}; i < size; ++i) {
            out.real[i] = noise_out_real(i);
            out.imag[i] = noise_out_imag(i);
        }
        state.ResumeTiming();

        neo::multiply_add(x, y, out, out);

        benchmark::DoNotOptimize(out.real[0]);
        benchmark::DoNotOptimize(out.imag[0]);
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * size;
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(Real) * 2);
}

template<typename FixedPointBatch>
auto batch_split_multiply_add(benchmark::State& state) -> void
{
    using FxPoint = typename FixedPointBatch::value_type;

    auto const size        = state.range(0);
    auto const num_batches = size / FixedPointBatch::size;

    auto x_buf   = stdex::mdarray<FixedPointBatch, stdex::dextents<size_t, 2>>{2, num_batches};
    auto y_buf   = stdex::mdarray<FixedPointBatch, stdex::dextents<size_t, 2>>{2, num_batches};
    auto out_buf = stdex::mdarray<FixedPointBatch, stdex::dextents<size_t, 2>>{2, num_batches};

    auto const x = neo::split_complex{
        stdex::submdspan(x_buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(x_buf.to_mdspan(), 1, stdex::full_extent),
    };
    auto const y = neo::split_complex{
        stdex::submdspan(y_buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(y_buf.to_mdspan(), 1, stdex::full_extent),
    };
    auto const out = neo::split_complex{
        stdex::submdspan(out_buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(out_buf.to_mdspan(), 1, stdex::full_extent),
    };

    auto const noise_x_real   = neo::generate_noise_signal<FxPoint>(num_batches, std::random_device{}());
    auto const noise_x_imag   = neo::generate_noise_signal<FxPoint>(num_batches, std::random_device{}());
    auto const noise_y_real   = neo::generate_noise_signal<FxPoint>(num_batches, std::random_device{}());
    auto const noise_y_imag   = neo::generate_noise_signal<FxPoint>(num_batches, std::random_device{}());
    auto const noise_out_real = neo::generate_noise_signal<FxPoint>(num_batches, std::random_device{}());
    auto const noise_out_imag = neo::generate_noise_signal<FxPoint>(num_batches, std::random_device{}());

    for (auto i{0U}; i < num_batches; ++i) {
        x.real[i] = FixedPointBatch::broadcast(noise_x_real(i));
        x.imag[i] = FixedPointBatch::broadcast(noise_x_imag(i));
        y.real[i] = FixedPointBatch::broadcast(noise_y_real(i));
        y.imag[i] = FixedPointBatch::broadcast(noise_y_imag(i));
    }

    for (auto _ : state) {
        state.PauseTiming();
        for (auto i{0U}; i < num_batches; ++i) {
            out.real[i] = FixedPointBatch::broadcast(noise_out_real(i));
            out.imag[i] = FixedPointBatch::broadcast(noise_out_imag(i));
        }
        state.ResumeTiming();

        neo::multiply_add(x, y, out, out);

        benchmark::DoNotOptimize(out.real[0]);
        benchmark::DoNotOptimize(out.imag[0]);
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * size;
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(FxPoint) * 2);
}

}  // namespace

BENCHMARK(multiply_add<std::complex<float>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
BENCHMARK(multiply_add<std::complex<double>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);

#if defined(NEO_HAS_BUILTIN_FLOAT16)
BENCHMARK(multiply_add<neo::complex32>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
#endif
BENCHMARK(multiply_add<neo::complex64>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
BENCHMARK(multiply_add<neo::complex128>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);

#if defined(NEO_HAS_BUILTIN_FLOAT16)
BENCHMARK(split_multiply_add<_Float16>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
#endif
BENCHMARK(split_multiply_add<float>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
BENCHMARK(split_multiply_add<double>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);

#if defined(NEO_HAS_XSIMD)
BENCHMARK(batch_split_multiply_add<xsimd::batch<float>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
BENCHMARK(batch_split_multiply_add<xsimd::batch<double>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
#endif

BENCHMARK(split_multiply_add<neo::q7>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
BENCHMARK(split_multiply_add<neo::q15>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);

#if defined(NEO_HAS_ISA_AVX2)
BENCHMARK(batch_split_multiply_add<neo::q15x8>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
BENCHMARK(batch_split_multiply_add<neo::q15x16>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
#endif

// BENCHMARK(multiply_add<float>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
// BENCHMARK(multiply_add<double>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);

// BENCHMARK(multiply_add<neo::q7>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
// BENCHMARK(multiply_add<neo::q15>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);

BENCHMARK_MAIN();
