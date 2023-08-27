#include <neo/algorithm.hpp>
#include <neo/fft.hpp>
#include <neo/simd.hpp>

#include <neo/testing/benchmark.hpp>
#include <neo/testing/testing.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <string_view>

namespace {

template<typename Func>
auto timeit(std::string_view name, size_t N, Func func)
{
    using microseconds = std::chrono::duration<double, std::micro>;

    auto const num_runs = 50'000ULL;
    auto const padding  = num_runs / 100ULL * 2ULL;

    auto all_runs = stdex::mdarray<double, stdex::dextents<size_t, 1>>{num_runs};

    for (auto i{0}; i < num_runs; ++i) {
        auto start = std::chrono::system_clock::now();
        func();
        auto stop = std::chrono::system_clock::now();

        all_runs(i) = std::chrono::duration_cast<microseconds>(stop - start).count();
    }

    std::sort(all_runs.data(), std::next(all_runs.data(), all_runs.size()));

    auto const runs   = stdex::submdspan(all_runs.to_mdspan(), std::tuple{padding, all_runs.extent(0) - padding});
    auto const dsize  = static_cast<double>(N);
    auto const avg    = neo::mean(runs).value();
    auto const stddev = neo::standard_deviation(runs).value();
    auto const mflops = static_cast<int>(std::lround(5.0 * dsize * std::log2(dsize) / avg)) * 2;

    std::printf(
        "%-32s N: %-5zu - runs: %zu - avg: %.1fus - stddev: %.1fus - min: %.1fus - max: %.1fus - mflops: %d\n",
        name.data(),
        N,
        all_runs.extent(0),
        avg,
        stddev,
        runs[0],
        runs[runs.extent(0) - 1U],
        mflops
    );
}

template<typename Plan>
struct fft_bench
{
    using complex_type = typename Plan::complex_type;
    using real_type    = typename complex_type::value_type;

    explicit fft_bench(size_t size)
        : _plan{neo::ilog2(size)}
        , _buffer(neo::generate_noise_signal<complex_type>(_plan.size(), std::random_device{}()))
    {}

    auto operator()() -> void
    {
        auto const buffer = _buffer.to_mdspan();

        neo::fft::fft(_plan, buffer);
        neo::fft::ifft(_plan, buffer);
        neo::scale(real_type(1) / static_cast<real_type>(_plan.size()), buffer);

        neo::do_not_optimize(buffer[0]);
        neo::do_not_optimize(buffer[buffer.extent(0) - 1]);
    }

private:
    Plan _plan;
    stdex::mdarray<complex_type, stdex::dextents<size_t, 1>> _buffer;
};

#if defined(NEO_PLATFORM_APPLE)

struct vdsp_fft_bench
{
    explicit vdsp_fft_bench(size_t size) : _size{size}, _input{2, _size}, _output{2, _size}
    {
        auto const noise = neo::generate_noise_signal<neo::complex64>(size, std::random_device{}());
        for (auto i{0U}; i < size; ++i) {
            _input(0, i) = noise(i).real();
            _input(1, i) = noise(i).imag();
        }
    }

    ~vdsp_fft_bench()
    {
        if (_plan != nullptr) {
            vDSP_destroy_fftsetup(_plan);
        }
    }

    auto operator()() -> void
    {
        auto in  = DSPSplitComplex{.realp = std::addressof(_input(0, 0)), .imagp = std::addressof(_input(1, 0))};
        auto out = DSPSplitComplex{.realp = std::addressof(_output(0, 0)), .imagp = std::addressof(_output(1, 0))};

        vDSP_fft_zop(_plan, &in, 1, &out, 1, _order, kFFTDirection_Forward);
        vDSP_fft_zop(_plan, &out, 1, &in, 1, _order, kFFTDirection_Inverse);

        auto const factor = 1.0F / static_cast<float>(_size);
        vDSP_vsmul(in.realp, 1, &factor, in.realp, 1, _size);
        vDSP_vsmul(in.imagp, 1, &factor, in.imagp, 1, _size);

        neo::do_not_optimize(in.imagp[0]);
        neo::do_not_optimize(in.realp[0]);
    }

private:
    size_t _size;
    vDSP_Length _order{static_cast<vDSP_Length>(neo::ilog2(_size))};
    FFTSetup _plan{vDSP_create_fftsetup(_order, 2)};
    stdex::mdarray<float, stdex::dextents<size_t, 2>> _input;
    stdex::mdarray<float, stdex::dextents<size_t, 2>> _output;
};

#endif

}  // namespace

auto main(int argc, char** argv) -> int
{
    using namespace neo::fft;

    auto N = 1024UL;
    if (argc == 2) {
        N = std::stoul(argv[1]);
    }

#if defined(NEO_PLATFORM_APPLE)
    timeit("apple_vdsp_parallel<float> ", N, vdsp_fft_bench{N});
    timeit("apple_vdsp<complex<float>> ", N, fft_bench<fft_apple_vdsp_plan<std::complex<float>>>{N});
    timeit("apple_vdsp<complex64>      ", N, fft_bench<fft_apple_vdsp_plan<neo::complex64>>{N});
    std::printf("\n");
#endif

    timeit("fft<complex<float>, v1>", N, fft_bench<fft_radix2_plan<std::complex<float>, radix2_kernel_v1>>{N});
    timeit("fft<complex<float>, v2>", N, fft_bench<fft_radix2_plan<std::complex<float>, radix2_kernel_v2>>{N});
    timeit("fft<complex<float>, v3>", N, fft_bench<fft_radix2_plan<std::complex<float>, radix2_kernel_v3>>{N});
    timeit("fft<complex<float>, v4>", N, fft_bench<fft_radix2_plan<std::complex<float>, radix2_kernel_v4>>{N});
    std::printf("\n");

    timeit("fft<complex64, v1>", N, fft_bench<fft_radix2_plan<neo::complex64, radix2_kernel_v1>>{N});
    timeit("fft<complex64, v2>", N, fft_bench<fft_radix2_plan<neo::complex64, radix2_kernel_v2>>{N});
    timeit("fft<complex64, v3>", N, fft_bench<fft_radix2_plan<neo::complex64, radix2_kernel_v3>>{N});
    timeit("fft<complex64, v4>", N, fft_bench<fft_radix2_plan<neo::complex64, radix2_kernel_v4>>{N});
    std::printf("\n");

    return EXIT_SUCCESS;
}
