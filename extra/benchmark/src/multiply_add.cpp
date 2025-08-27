// SPDX-License-Identifier: MIT

#include "multi_batch.hpp"

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

template<typename Batch>
auto batch_split_multiply_add(benchmark::State& state) -> void
{
    using Real = typename Batch::value_type;

    auto const size = state.range(0);

    auto x_buf   = stdex::mdarray<Batch, stdex::dextents<size_t, 2>>{2, size};
    auto y_buf   = stdex::mdarray<Batch, stdex::dextents<size_t, 2>>{2, size};
    auto out_buf = stdex::mdarray<Batch, stdex::dextents<size_t, 2>>{2, size};

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
        x.real[i] = Batch::broadcast(noise_x_real(i));
        x.imag[i] = Batch::broadcast(noise_x_imag(i));
        y.real[i] = Batch::broadcast(noise_y_real(i));
        y.imag[i] = Batch::broadcast(noise_y_imag(i));
    }

    for (auto _ : state) {
        state.PauseTiming();
        for (auto i{0U}; i < size; ++i) {
            out.real[i] = Batch::broadcast(noise_out_real(i));
            out.imag[i] = Batch::broadcast(noise_out_imag(i));
        }
        state.ResumeTiming();

        neo::multiply_add(x, y, out, out);

        benchmark::DoNotOptimize(out.real[0]);
        benchmark::DoNotOptimize(out.imag[0]);
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * size * Batch::size;
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(Real) * 2);
}

template<typename MultiBatch>
auto multi_batch_split_multiply_add(benchmark::State& state) -> void
{
    using Batch = typename MultiBatch::batch_type;
    using Real  = typename MultiBatch::real_type;

    auto const size = state.range(0);

    auto x_buf   = stdex::mdarray<MultiBatch, stdex::dextents<size_t, 2>>{2, size};
    auto y_buf   = stdex::mdarray<MultiBatch, stdex::dextents<size_t, 2>>{2, size};
    auto out_buf = stdex::mdarray<MultiBatch, stdex::dextents<size_t, 2>>{2, size};

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
        x.real[i] = Batch::broadcast(noise_x_real(i));
        x.imag[i] = Batch::broadcast(noise_x_imag(i));
        y.real[i] = Batch::broadcast(noise_y_real(i));
        y.imag[i] = Batch::broadcast(noise_y_imag(i));
    }

    for (auto _ : state) {
        state.PauseTiming();
        for (auto i{0U}; i < size; ++i) {
            out.real[i] = Batch::broadcast(noise_out_real(i));
            out.imag[i] = Batch::broadcast(noise_out_imag(i));
        }
        state.ResumeTiming();

        neo::multiply_add(x, y, out, out);

        benchmark::DoNotOptimize(out.real[0]);
        benchmark::DoNotOptimize(out.imag[0]);
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * size * MultiBatch::size;
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(Real) * 2);
}

}  // namespace

// BENCHMARK(multiply_add<std::complex<float>>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24);
// BENCHMARK(multiply_add<std::complex<double>>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24);

#if defined(NEO_HAS_BUILTIN_FLOAT16)
// BENCHMARK(multiply_add<neo::complex32>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("complex32");
#endif
// BENCHMARK(multiply_add<neo::complex64>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("complex64");
// BENCHMARK(multiply_add<neo::complex128>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("complex128");

#if defined(NEO_HAS_BUILTIN_FLOAT16)
// BENCHMARK(split_multiply_add<_Float16>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("split_complex<_Float16>");
#endif
// BENCHMARK(split_multiply_add<float>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("split_complex<float>");
// BENCHMARK(split_multiply_add<double>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("split_complex<double>");

#if defined(NEO_HAS_XSIMD)
// BENCHMARK(batch_split_multiply_add<xsimd::batch<float>>)->RangeMultiplier(4)->Range(1 << 7, 1 <<
// 24)->Name("split_complex<xsimd<float>>"); BENCHMARK(multi_batch_split_multiply_add<neo::multi_batch<float,
// 2>>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("split_complex<multi_batch<float, 2>>");
// BENCHMARK(multi_batch_split_multiply_add<neo::multi_batch<float, 4>>)->RangeMultiplier(4)->Range(1 << 7, 1 <<
// 24)->Name("split_complex<multi_batch<float, 4>>"); BENCHMARK(multi_batch_split_multiply_add<neo::multi_batch<float,
// 8>>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("split_complex<multi_batch<float, 8>>");
// BENCHMARK(batch_split_multiply_add<xsimd::batch<double>>)->RangeMultiplier(4)->Range(1 << 7, 1 <<
// 24)->Name("split_complex<xsimd<double>>");
#endif

BENCHMARK(split_multiply_add<neo::q7>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("q7");
BENCHMARK(split_multiply_add<neo::q15>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("q15");

#if defined(NEO_HAS_ISA_NEON) or defined(NEO_HAS_ISA_AVX2)
BENCHMARK(batch_split_multiply_add<neo::q15x8>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("q15x8");
#endif

#if defined(NEO_HAS_ISA_AVX2)
BENCHMARK(batch_split_multiply_add<neo::q15x16>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24)->Name("q15x16");
#endif

// BENCHMARK(multiply_add<float>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24);
// BENCHMARK(multiply_add<double>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24);

// BENCHMARK(multiply_add<neo::q7>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24);
// BENCHMARK(multiply_add<neo::q15>)->RangeMultiplier(4)->Range(1 << 7, 1 << 24);

BENCHMARK_MAIN();
