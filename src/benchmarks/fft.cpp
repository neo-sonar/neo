#include <neo/fft.hpp>
#include <neo/simd.hpp>

#include <neo/testing/benchmark.hpp>
#include <neo/testing/testing.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string_view>
#include <vector>

namespace {

template<typename Func>
auto timeit(std::string_view name, size_t N, Func func)
{
    using microseconds = std::chrono::duration<double, std::micro>;

    auto const num_runs = 50'000ULL;
    auto const padding  = num_runs / 100ULL * 2ULL;

    auto all_runs = std::vector<double>{};
    all_runs.reserve(num_runs);

    for (auto i{0}; i < num_runs; ++i) {
        auto start = std::chrono::system_clock::now();
        func();
        auto stop = std::chrono::system_clock::now();

        all_runs.push_back(std::chrono::duration_cast<microseconds>(stop - start).count());
    }

    std::sort(all_runs.begin(), all_runs.end());

    auto const runs   = std::span<double>(all_runs).subspan(padding, all_runs.size() - padding * 2);
    auto const dsize  = static_cast<double>(N);
    auto const avg    = std::reduce(runs.begin(), runs.end(), 0.0) / double(runs.size());
    auto const mflops = static_cast<int>(std::lround(5.0 * dsize * std::log2(dsize) / avg)) * 2;

    std::printf(
        "%-32s N: %-5zu - runs: %zu - avg: %.1fus - min: %.1fus - max: "
        "%.1fus - mflops: %d\n",
        name.data(),
        N,
        std::size(all_runs),
        avg,
        *std::min_element(runs.begin(), runs.end()),
        *std::max_element(runs.begin(), runs.end()),
        mflops
    );
}

template<typename Complex, typename Kernel>
struct fft_roundtrip
{
    explicit fft_roundtrip(size_t size)
        : _buf(neo::generate_noise_signal<Complex>(size, std::random_device{}()))
        , _fft{neo::ilog2(size)}
    {}

    auto operator()() -> void
    {
        auto const buf = _buf.to_mdspan();
        _fft(buf, neo::fft::direction::forward);
        _fft(buf, neo::fft::direction::backward);
        neo::do_not_optimize(buf[0]);
        neo::do_not_optimize(buf[buf.extent(0) - 1]);
    }

private:
    stdex::mdarray<Complex, stdex::dextents<size_t, 1>> _buf;
    neo::fft::fft_radix2_plan<Complex, Kernel> _fft;
};

}  // namespace

namespace fft = neo::fft;

auto main() -> int
{
    // auto tw_storage = neo::fft::make_radix2_twiddles<std::complex<double>>(8);
    // auto tw         = std::span{tw_storage.data(), tw_storage.extent(0)};
    // for (auto i{0U}; i < tw.size(); ++i) {
    //     std::cout << i << ": " << tw[i] << '\n';
    // }

    static constexpr auto N = 1024;

    timeit("fft<std::complex<float>, v1>(N)", N, fft_roundtrip<std::complex<float>, fft::radix2_kernel_v1>{N});
    timeit("fft<std::complex<float>, v2>(N)", N, fft_roundtrip<std::complex<float>, fft::radix2_kernel_v2>{N});
    timeit("fft<std::complex<float>, v3>(N)", N, fft_roundtrip<std::complex<float>, fft::radix2_kernel_v3>{N});
    timeit("fft<std::complex<float>, v4>(N)", N, fft_roundtrip<std::complex<float>, fft::radix2_kernel_v4>{N});
    std::printf("\n");

    timeit("fft<neo::complex64, v1>(N)", N, fft_roundtrip<neo::complex64, fft::radix2_kernel_v1>{N});
    timeit("fft<neo::complex64, v2>(N)", N, fft_roundtrip<neo::complex64, fft::radix2_kernel_v2>{N});
    timeit("fft<neo::complex64, v3>(N)", N, fft_roundtrip<neo::complex64, fft::radix2_kernel_v3>{N});
    timeit("fft<neo::complex64, v4>(N)", N, fft_roundtrip<neo::complex64, fft::radix2_kernel_v4>{N});
    std::printf("\n");

    return EXIT_SUCCESS;
}
