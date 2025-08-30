// SPDX-License-Identifier: MIT

#include "multi_batch.hpp"

#include <neo/fft.hpp>

#include <neo/config/xsimd.hpp>
#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

template<typename>
struct float_type;

template<typename T>
struct float_type<xsimd::batch<T>>
{
    using type = neo::value_type_t<xsimd::batch<T>>;
};

template<typename T, std::size_t Size>
struct float_type<neo::multi_batch<T, Size>>
{
    using type = typename neo::multi_batch<T, Size>::real_type;
};

template<typename T>
using float_type_t = typename float_type<T>::type;

template<typename Complex, typename Kernel>
struct simd_fft_plan
{
    using value_type = xsimd::batch<Complex>;
    using size_type  = std::size_t;

    simd_fft_plan(neo::fft::from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{27}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return neo::ipow<2zu>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<neo::inout_vector Vec>
        requires std::same_as<neo::value_type_t<Vec>, value_type>
    auto operator()(Vec x, neo::fft::direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.extent(0), size()));

        _reorder(x);

        auto const kernel = Kernel{};
        if (dir == neo::fft::direction::forward) {
            kernel(x, _twiddles.to_mdspan());
        } else {
            // kernel(x, conjugate_view{_twiddles.to_mdspan()});
        }
    }

private:
    static auto twiddle(size_type n) -> stdex::mdarray<value_type, stdex::dextents<size_t, 1>>
    {
        using scalar_type = neo::value_type_t<value_type>;
        auto const scalar = neo::fft::make_twiddle_lut_radix2<scalar_type>(n, neo::fft::direction::forward);
        auto vec          = stdex::mdarray<value_type, stdex::dextents<size_t, 1>>{scalar.extent(0)};
        for (auto i{0U}; i < scalar.extent(0); ++i) {
            vec(i) = xsimd::broadcast(scalar(i));
        }
        return vec;
    }

    size_type _order;
    neo::fft::bitrevorder_plan _reorder{static_cast<size_t>(_order)};
    stdex::mdarray<value_type, stdex::dextents<size_type, 1>> _twiddles{twiddle(size())};
};

template<typename FloatBatch>
struct simd_split_fft_plan
{
    using value_type = FloatBatch;
    using size_type  = std::size_t;

    simd_split_fft_plan(neo::fft::from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<neo::inout_vector_of<FloatBatch> InOutVec>
    auto operator()(neo::split_complex<InOutVec> x, neo::fft::direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.real.extent(0), size()));
        assert(neo::detail::extents_equal(x.real, x.imag));

        _reorder(x);

        if (dir == neo::fft::direction::forward) {
            stage_0(x.real, x.imag);
        } else {
            stage_0(x.imag, x.real);
        }
    }

    template<neo::in_vector_of<FloatBatch> InVec, neo::out_vector_of<FloatBatch> OutVec>
    auto operator()(neo::split_complex<InVec> in, neo::split_complex<OutVec> out, neo::fft::direction dir) noexcept
        -> void
    {
        assert(std::cmp_equal(in.real.extent(0), size()));
        assert(neo::detail::extents_equal(in.real, in.imag, out.real, out.imag));

        copy(in.real, out.real);
        copy(in.imag, out.imag);
        (*this)(out, dir);
    }

private:
    auto stage_0(neo::inout_vector_of<FloatBatch> auto xre, neo::inout_vector_of<FloatBatch> auto xim) -> void
    {
        static constexpr auto const stage_length = 1;  // ipow<2>(0)
        static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

        for (auto k{0}; k < static_cast<int>(size()); k += stride) {
            auto const i1 = k;
            auto const i2 = k + stage_length;

            auto const x1re = xre[i1];
            auto const x1im = xim[i1];
            auto const x2re = xre[i2];
            auto const x2im = xim[i2];

            xre[i1] = x1re + x2re;
            xim[i1] = x1im + x2im;

            xre[i2] = x1re - x2re;
            xim[i2] = x1im - x2im;
        }

        auto const tw_re = stdex::submdspan(_tw.to_mdspan(), 0, stdex::full_extent);
        auto const tw_im = stdex::submdspan(_tw.to_mdspan(), 1, stdex::full_extent);
        stage_n(xre, xim, tw_re, tw_im);
    }

    auto stage_n(
        neo::inout_vector_of<FloatBatch> auto xre,
        neo::inout_vector_of<FloatBatch> auto xim,
        neo::in_vector_of<FloatBatch> auto tw_re,
        neo::in_vector_of<FloatBatch> auto tw_im
    ) -> void
    {
        auto const log2_size = static_cast<int>(order());
        auto const size      = 1 << log2_size;

        for (auto stage{1}; stage < log2_size; ++stage) {

            auto const stage_length = neo::ipow<2>(stage);
            auto const stride       = neo::ipow<2>(stage + 1);
            auto const tw_stride    = neo::ipow<2>(log2_size - stage - 1);

            for (auto k{0}; k < size; k += stride) {
                for (auto pair{0}; pair < stage_length; ++pair) {
                    auto const i1      = k + pair;
                    auto const i2      = k + pair + stage_length;
                    auto const w_index = pair * tw_stride;

                    auto const wre = tw_re[w_index];
                    auto const wim = tw_im[w_index];

                    auto const x1re = xre[i1];
                    auto const x1im = xim[i1];
                    auto const x2re = xre[i2];
                    auto const x2im = xim[i2];

                    auto const [xwre, xwim] = [&] {
                        if constexpr (xsimd::is_batch<FloatBatch>::value) {
                            auto const re = xsimd::fms(wre, x2re, wim * x2im);
                            auto const im = xsimd::fma(wre, x2im, wim * x2re);
                            return std::pair{re, im};
                        } else {
                            auto const re = fms(wre, x2re, wim * x2im);
                            auto const im = fma(wre, x2im, wim * x2re);
                            return std::pair{re, im};
                        }
                    }();

                    xre[i1] = x1re + xwre;
                    xim[i1] = x1im + xwim;
                    xre[i2] = x1re - xwre;
                    xim[i2] = x1im - xwim;
                }
            }
        }
    }

    [[nodiscard]] static auto twiddle(size_type n)
    {
        using scalar_type = float_type_t<FloatBatch>;
        using complex     = std::complex<scalar_type>;

        auto const dir         = neo::fft::direction::forward;
        auto const interleaved = neo::fft::make_twiddle_lut_radix2<complex>(n, dir);

        auto w_buf   = stdex::mdarray<FloatBatch, stdex::dextents<size_t, 2>>{2, interleaved.extent(0)};
        auto w_re    = stdex::submdspan(w_buf.to_mdspan(), 0, stdex::full_extent);
        auto w_im    = stdex::submdspan(w_buf.to_mdspan(), 1, stdex::full_extent);
        auto split_w = neo::split_complex{w_re, w_im};

        for (auto i{0U}; i < interleaved.extent(0); ++i) {
            auto const w    = interleaved(i);
            split_w.real[i] = xsimd::broadcast(neo::math::real(w));
            split_w.imag[i] = xsimd::broadcast(neo::math::imag(w));
        }

        return w_buf;
    }

    size_type _order;
    neo::fft::bitrevorder_plan _reorder{static_cast<size_t>(_order)};
    stdex::mdarray<FloatBatch, stdex::dextents<size_t, 2>> _tw{twiddle(size())};
};

template<typename ScalarComplex, typename Kernel>
auto simd_c2c(benchmark::State& state) -> void
{
    using Plan        = simd_fft_plan<ScalarComplex, Kernel>;
    using SimdComplex = neo::value_type_t<Plan>;

    auto len   = static_cast<std::size_t>(state.range(0));
    auto order = neo::fft::next_order(len);
    auto plan  = Plan{neo::fft::from_order, order};

    auto const noise_s = neo::generate_noise_signal<ScalarComplex>(len, std::random_device{}());
    auto noise_v       = stdex::mdarray<SimdComplex, stdex::dextents<size_t, 1>>{noise_s.extent(0)};
    for (auto i{0U}; i < noise_s.extent(0); ++i) {
        noise_v(i) = xsimd::broadcast(noise_s(i));
    }

    auto work = noise_v;

    for (auto _ : state) {
        state.PauseTiming();
        neo::copy(noise_v.to_mdspan(), work.to_mdspan());
        state.ResumeTiming();

        neo::fft::fft(plan, work.to_mdspan());

        benchmark::DoNotOptimize(work(0));
        benchmark::ClobberMemory();
    }

    auto const items       = static_cast<int64_t>(state.iterations()) * plan.size();
    auto const flop        = 5zu * plan.order() * items * SimdComplex::size;
    state.counters["flop"] = benchmark::Counter(static_cast<double>(flop), benchmark::Counter::kIsRate);
    state.SetBytesProcessed(items * sizeof(SimdComplex));
}

template<typename SimdFloat>
auto simd_split_c2c(benchmark::State& state) -> void
{
    using Plan  = simd_split_fft_plan<SimdFloat>;
    using Float = float_type_t<SimdFloat>;

    auto const len   = static_cast<std::size_t>(state.range(0));
    auto const order = neo::fft::next_order(len);
    auto plan        = Plan{neo::fft::from_order, order};

    auto const noise = neo::generate_noise_signal<Float>(len, std::random_device{}());
    auto noise_v     = stdex::mdarray<SimdFloat, stdex::dextents<size_t, 2>>{2, noise.extent(0)};
    auto noise_r     = stdex::submdspan(noise_v.to_mdspan(), 0, stdex::full_extent);
    for (auto i{0U}; i < noise.extent(0); ++i) {
        noise_r[i] = xsimd::broadcast(noise(i));
    }

    auto buf = stdex::mdarray<SimdFloat, stdex::dextents<std::size_t, 2>>{2, len};
    auto z   = neo::split_complex{
        stdex::submdspan(buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(buf.to_mdspan(), 1, stdex::full_extent),
    };

    for (auto _ : state) {
        state.PauseTiming();
        neo::copy(noise_r, z.real);
        neo::fill(z.imag, Float(0));
        state.ResumeTiming();

        plan(z, neo::fft::direction::forward);

        benchmark::DoNotOptimize(z.real[0]);
        benchmark::DoNotOptimize(z.imag[0]);
        benchmark::ClobberMemory();
    }

    auto const items       = static_cast<int64_t>(state.iterations()) * plan.size();
    auto const flop        = 5zu * plan.order() * items * SimdFloat::size;
    state.counters["flop"] = benchmark::Counter(static_cast<double>(flop), benchmark::Counter::kIsRate);
    state.SetBytesProcessed(items * sizeof(SimdFloat) * 2);
}

}  // namespace

BENCHMARK(simd_c2c<std::complex<float>, neo::fft::kernel::c2c_dit2_v1>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 16)
    ->Name("c2c_dit2_v1");
BENCHMARK(simd_c2c<std::complex<float>, neo::fft::kernel::c2c_dit2_v2>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 16)
    ->Name("c2c_dit2_v2");
BENCHMARK(simd_c2c<std::complex<float>, neo::fft::kernel::c2c_dit2_v3>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 16)
    ->Name("c2c_dit2_v3");
BENCHMARK(simd_c2c<std::complex<float>, neo::fft::kernel::c2c_dit2_v4>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 16)
    ->Name("c2c_dit2_v4");
BENCHMARK(simd_split_c2c<xsimd::batch<float>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 16)->Name("xsimd::batch");
BENCHMARK(simd_split_c2c<neo::multi_batch<float, 2>>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 16)
    ->Name("multi_batch<2>");
BENCHMARK(simd_split_c2c<neo::multi_batch<float, 4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 16)
    ->Name("multi_batch<4>");

BENCHMARK_MAIN();
