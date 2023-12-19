#include <neo/algorithm.hpp>
#include <neo/fft.hpp>
#include <neo/simd.hpp>

#include <neo/testing/benchmark.hpp>
#include <neo/testing/testing.hpp>

#include <fmt/format.h>
#include <fmt/os.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <string_view>

namespace {

template<typename Func>
auto timeit(std::string_view name, size_t n, Func func)
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

    std::sort(all_runs.data(), std::next(all_runs.data(), ptrdiff_t(all_runs.size())));

    auto const runs   = stdex::submdspan(all_runs.to_mdspan(), std::tuple{padding, all_runs.extent(0) - padding});
    auto const dsize  = static_cast<double>(n);
    auto const avg    = neo::mean(runs);
    auto const stddev = neo::standard_deviation(runs);
    if (not avg or not stddev) {
        std::puts("failed benchmark\n");
        return;
    }

    auto const mflops = static_cast<int>(std::lround(5.0 * dsize * std::log2(dsize) / *avg)) * 2;

    fmt::println(
        "{:<32} N: {:<5} - runs: {} - avg: {:.1f}us - stddev: {:.1f}us - min: {:.1f}us - max: {:.1f}us - mflops: {}",
        name.data(),
        n,
        all_runs.extent(0),
        *avg,
        *stddev,
        runs[0],
        runs[runs.extent(0) - 1U],
        mflops
    );
}

template<typename Plan>
struct fft_bench
{
    using value_type = typename Plan::value_type;
    using real_type  = typename value_type::value_type;

    explicit fft_bench(size_t size)
        : _plan{neo::ilog2(size)}
        , _buffer(neo::generate_noise_signal<value_type>(_plan.size(), std::random_device{}()))
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
    stdex::mdarray<value_type, stdex::dextents<size_t, 1>> _buffer;
};

template<typename Plan>
struct fftr_bench
{
    using real_type = typename Plan::value_type;

    explicit fftr_bench(size_t size)
        : _plan{neo::ilog2(size)}
        , _buffer(neo::generate_noise_signal<real_type>(_plan.size() * 2U, std::random_device{}()))
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
    stdex::mdarray<real_type, stdex::dextents<size_t, 1>> _buffer;
};

#if defined(NEO_PLATFORM_APPLE)

template<std::floating_point Float>
struct vdsp_fft_bench
{
    using native_handle_type = std::conditional_t<std::same_as<Float, float>, FFTSetup, FFTSetupD>;
    using split_complex_type = std::conditional_t<std::same_as<Float, float>, DSPSplitComplex, DSPDoubleSplitComplex>;

    explicit vdsp_fft_bench(size_t size) : _size{size}, _input{2, _size}, _output{2, _size}
    {
        auto const noise = neo::generate_noise_signal<std::complex<Float>>(size, std::random_device{}());
        for (auto i{0U}; i < size; ++i) {
            _input(0, i) = noise(i).real();
            _input(1, i) = noise(i).imag();
        }
    }

    ~vdsp_fft_bench()
    {
        if (_plan != nullptr) {
            if constexpr (std::same_as<Float, float>) {
                vDSP_destroy_fftsetup(_plan);
            } else {
                vDSP_destroy_fftsetupD(_plan);
            }
        }
    }

    auto operator()() -> void
    {
        auto in  = split_complex_type{.realp = std::addressof(_input(0, 0)), .imagp = std::addressof(_input(1, 0))};
        auto out = split_complex_type{.realp = std::addressof(_output(0, 0)), .imagp = std::addressof(_output(1, 0))};

        auto const factor = Float(1) / static_cast<Float>(_size);
        auto const stride = 1;

        if constexpr (std::same_as<Float, float>) {
            vDSP_fft_zop(_plan, &in, stride, &out, stride, _order, kFFTDirection_Forward);
            vDSP_fft_zop(_plan, &out, stride, &in, stride, _order, kFFTDirection_Inverse);
            vDSP_vsmul(in.realp, stride, &factor, in.realp, stride, _size);
            vDSP_vsmul(in.imagp, stride, &factor, in.imagp, stride, _size);
        } else {
            vDSP_fft_zopD(_plan, &in, stride, &out, stride, _order, kFFTDirection_Forward);
            vDSP_fft_zopD(_plan, &out, stride, &in, stride, _order, kFFTDirection_Inverse);
            vDSP_vsmulD(in.realp, stride, &factor, in.realp, stride, _size);
            vDSP_vsmulD(in.imagp, stride, &factor, in.imagp, stride, _size);
        }

        neo::do_not_optimize(in.imagp[0]);
        neo::do_not_optimize(in.realp[0]);
        neo::do_not_optimize(out.imagp[0]);
        neo::do_not_optimize(out.realp[0]);
    }

private:
    size_t _size;
    vDSP_Length _order{static_cast<vDSP_Length>(neo::ilog2(_size))};
    native_handle_type _plan{[this] {
        if constexpr (std::same_as<Float, float>) {
            return vDSP_create_fftsetup(_order, 2);
        } else {
            return vDSP_create_fftsetupD(_order, 2);
        }
    }()};

    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _input;
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _output;
};

#endif

}  // namespace

auto main(int argc, char** argv) -> int
{
    using namespace neo::fft;

    auto const args = std::span<char const* const>{argv, size_t(argc)};

    auto n = 1024UL;
    if (args.size() == 2) {
        n = std::stoul(args[1]);
    }

#if defined(NEO_PLATFORM_APPLE)
    timeit("apple_vdsp_parallel<float>  ", n, vdsp_fft_bench<float>{n});
    timeit("apple_vdsp<complex<float>>  ", n, fft_bench<fft_apple_vdsp_plan<std::complex<float>>>{n});
    timeit("apple_vdsp<complex64>       ", n, fft_bench<fft_apple_vdsp_plan<neo::complex64>>{n});
    std::puts("\n");
#endif

    timeit("fft<complex<float>, v1>", n, fft_bench<fft_plan<std::complex<float>, kernel::c2c_dit2_v1>>{n});
    timeit("fft<complex<float>, v2>", n, fft_bench<fft_plan<std::complex<float>, kernel::c2c_dit2_v2>>{n});
    timeit("fft<complex<float>, v3>", n, fft_bench<fft_plan<std::complex<float>, kernel::c2c_dit2_v3>>{n});
    std::puts("\n");

    timeit("fft<complex64, v1>", n, fft_bench<fft_plan<neo::complex64, kernel::c2c_dit2_v1>>{n});
    timeit("fft<complex64, v2>", n, fft_bench<fft_plan<neo::complex64, kernel::c2c_dit2_v2>>{n});
    timeit("fft<complex64, v3>", n, fft_bench<fft_plan<neo::complex64, kernel::c2c_dit2_v3>>{n});
    std::puts("\n");

    timeit("fftr<float>", n, fftr_bench<experimental::fft_plan<float>>{n});
    std::puts("\n");

#if defined(NEO_PLATFORM_APPLE)
    timeit("apple_vdsp_parallel<double> ", n, vdsp_fft_bench<double>{n});
    timeit("apple_vdsp<complex<double>> ", n, fft_bench<fft_apple_vdsp_plan<std::complex<double>>>{n});
    timeit("apple_vdsp<complex128>      ", n, fft_bench<fft_apple_vdsp_plan<neo::complex128>>{n});
    std::puts("\n");
#endif

    timeit("fft<complex<double>, v1>", n, fft_bench<fft_plan<std::complex<double>, kernel::c2c_dit2_v1>>{n});
    timeit("fft<complex<double>, v2>", n, fft_bench<fft_plan<std::complex<double>, kernel::c2c_dit2_v2>>{n});
    timeit("fft<complex<double>, v3>", n, fft_bench<fft_plan<std::complex<double>, kernel::c2c_dit2_v3>>{n});
    std::puts("\n");

    timeit("fft<complex128, v1>", n, fft_bench<fft_plan<neo::complex128, kernel::c2c_dit2_v1>>{n});
    timeit("fft<complex128, v2>", n, fft_bench<fft_plan<neo::complex128, kernel::c2c_dit2_v2>>{n});
    timeit("fft<complex128, v3>", n, fft_bench<fft_plan<neo::complex128, kernel::c2c_dit2_v3>>{n});
    std::puts("\n");

    timeit("fftr<double>", n, fftr_bench<experimental::fft_plan<double>>{n});
    std::puts("\n");

    return EXIT_SUCCESS;
}