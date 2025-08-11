#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <numbers>
#include <numeric>
#include <random>
#include <span>
#include <string_view>
#include <vector>

template<typename Func>
auto timeit(std::string_view name, size_t N, Func func)
{
    using microseconds = std::chrono::duration<double, std::micro>;

    auto const size       = N;
    auto const iterations = 50'000U;
    auto const margin     = iterations / 30U;

    auto all_runs = std::vector<double>(iterations);

    func();
    func();
    func();

    for (auto i{0U}; i < iterations; ++i) {
        auto start = std::chrono::system_clock::now();
        func();
        auto stop = std::chrono::system_clock::now();

        all_runs[i] = std::chrono::duration_cast<microseconds>(stop - start).count();
    }

    // std::sort(all_runs.begin(), all_runs.end());
    auto const runs   = std::span{all_runs}.subspan(margin, all_runs.size() - margin * 2);
    auto const avg    = std::reduce(runs.begin(), runs.end(), 0.0) / double(runs.size());
    auto const dsize  = double(size);
    auto const mflops = static_cast<int>(std::lround(5.0 * dsize * std::log2(dsize) / avg)) * 2;

    std::printf("%-20s N: %zu - avg: %.1fus - mflops: %d\n", name.data(), N, avg, mflops);
}

template<typename Complex, typename URNG = std::mt19937>
[[nodiscard]] auto generate_noise_signal(std::size_t length, typename URNG::result_type seed)
{
    using Float = typename Complex::value_type;

    auto rng    = URNG{seed};
    auto dist   = std::uniform_real_distribution<Float>{Float(-1), Float(1)};
    auto signal = std::vector<Complex>(length);

    std::generate_n(signal.data(), signal.size(), [&] {
        return Complex{static_cast<Float>(dist(rng)), static_cast<Float>(dist(rng))};
    });

    return signal;
}

template<std::integral Int>
[[nodiscard]] constexpr auto bit_log2(Int x) -> Int
{
    auto result = Int{0};
    for (; x > Int(1); x >>= Int(1)) {
        ++result;
    }
    return result;
}

template<std::integral T>
[[nodiscard]] constexpr auto ipow(T base, T exponent) -> T
{
    T result = 1;
    for (T i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}

template<auto Base>
    requires std::integral<decltype(Base)>
[[nodiscard]] constexpr auto ipow(decltype(Base) exponent) -> decltype(Base)
{
    using Int = decltype(Base);

    if constexpr (Base == Int(2)) {
        return static_cast<Int>(Int(1) << exponent);
    } else {
        return ipow(Base, exponent);
    }
}

enum struct direction
{
    forward,
    backward,
};

template<typename Float>
auto fill_twiddle_lut_radix2(std::span<std::complex<Float>> table, direction dir = direction::forward) noexcept -> void
{
    auto const tableSize = table.size();
    auto const fftSize   = tableSize * 2ULL;
    auto const sign      = dir == direction::forward ? Float(-1) : Float(1);
    auto const twoPi     = static_cast<Float>(std::numbers::pi * 2.0);

    for (std::size_t i = 0; i < tableSize; ++i) {
        auto const angle = sign * twoPi * Float(i) / Float(fftSize);
        table[i]         = std::polar(Float(1), angle);
    }
}

template<typename Float>
auto make_twiddle_lut_radix2(std::size_t size, direction dir = direction::forward)
{
    auto table = std::vector<std::complex<Float>>(size / 2U);
    fill_twiddle_lut_radix2<Float>(table, dir);
    return table;
}

template<typename ElementType, typename IndexTable>
auto bitrevorder(std::span<ElementType> x, IndexTable const& index) -> void
{
    for (auto i{0U}; i < x.size(); ++i) {
        if (i < index[i]) {
            std::swap(x[i], x[index[i]]);
        }
    }
}

template<typename ElementType, typename IndexTable>
auto bitrevorder(std::span<ElementType> xre, std::span<ElementType> xim, IndexTable const& index) -> void
{
    for (auto i{0U}; i < xre.size(); ++i) {
        if (i < index[i]) {
            std::swap(xre[i], xre[index[i]]);
            std::swap(xim[i], xim[index[i]]);
        }
    }
}

[[nodiscard]] inline auto make_bitrevorder_table(std::size_t size) -> std::vector<std::size_t>
{
    auto const order = bit_log2(size);
    auto table       = std::vector<std::size_t>(size, 0);
    for (auto i{0U}; i < size; ++i) {
        for (auto j{0U}; j < order; ++j) {
            table[i] |= ((i >> j) & 1) << (order - 1 - j);
        }
    }
    return table;
}

template<typename Float, int Order, int Stage>
struct static_dit2_stage
{
    auto operator()(std::complex<Float>* __restrict__ x, std::complex<Float> const* __restrict__ twiddles) -> void
        requires(Stage == 0)
    {
        static constexpr auto const size         = 1 << Order;
        static constexpr auto const stage_length = 1;  // ipow<2>(0)
        static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

        for (auto k{0}; k < static_cast<int>(size); k += stride) {
            auto const i1 = k;
            auto const i2 = k + stage_length;

            auto const temp = x[i1] + x[i2];
            x[i2]           = x[i1] - x[i2];
            x[i1]           = temp;
        }

        static_dit2_stage<Float, Order, 1>{}(x, twiddles);
    }

    auto operator()(std::complex<Float>* __restrict__ x, std::complex<Float> const* __restrict__ twiddles) -> void
        requires(Stage != 0 and Stage < Order)
    {
        static constexpr auto const size         = 1 << Order;
        static constexpr auto const stage_length = ipow<2>(Stage);
        static constexpr auto const stride       = ipow<2>(Stage + 1);
        static constexpr auto const tw_stride    = ipow<2>(Order - Stage - 1);

        for (auto k{0}; k < size; k += stride) {
            for (auto pair{0}; pair < stage_length; ++pair) {
                auto const tw = twiddles[pair * tw_stride];

                auto const i1 = k + pair;
                auto const i2 = k + pair + stage_length;

                auto const temp = x[i1] + tw * x[i2];
                x[i2]           = x[i1] - tw * x[i2];
                x[i1]           = temp;
            }
        }

        static_dit2_stage<Float, Order, Stage + 1>{}(x, twiddles);
    }

    auto operator()(std::complex<Float>* __restrict__ /*x*/, std::complex<Float> const* __restrict__ /*twiddles*/)
        -> void
        requires(Stage == Order)
    {}
};

template<typename Float, int Order>
struct static_fft_plan
{
    static_fft_plan() = default;

    [[nodiscard]] static constexpr auto size() { return 1 << Order; }

    [[nodiscard]] static constexpr auto order() { return Order; }

    auto operator()(std::span<std::complex<Float>> x, direction dir) -> void
    {
        bitrevorder(x, _rev);

        if (dir == direction::forward) {
            static_dit2_stage<Float, Order, 0>{}(x.data(), _wf.data());
        } else {
            static_dit2_stage<Float, Order, 0>{}(x.data(), _wb.data());
        }
    }

private:
    std::vector<std::complex<Float>> _wf{make_twiddle_lut_radix2<Float>(size_t(size()), direction::forward)};
    std::vector<std::complex<Float>> _wb{make_twiddle_lut_radix2<Float>(size_t(size()), direction::backward)};
    std::vector<std::size_t> _rev{make_bitrevorder_table(size_t(size()))};
};

template<typename Float, int Order, int Stage>
struct split_fft_radix2_dit
{
    auto operator()(
        Float* __restrict__ xre,
        Float* __restrict__ xim,
        Float const* __restrict__ wre,
        Float const* __restrict__ wim
    ) -> void
        requires(Stage == 0)
    {
        static constexpr auto const size         = 1 << Order;
        static constexpr auto const stage_length = 1;  // ipow<2>(0)
        static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

        for (auto k{0}; k < static_cast<int>(size); k += stride) {
            auto const i1 = k;
            auto const i2 = k + stage_length;

            auto const x1 = std::complex{xre[i1], xim[i1]};
            auto const x2 = std::complex{xre[i2], xim[i2]};

            auto const xn1 = x1 + x2;
            xre[i1]        = xn1.real();
            xim[i1]        = xn1.imag();

            auto const xn2 = x1 - x2;
            xre[i2]        = xn2.real();
            xim[i2]        = xn2.imag();
        }

        split_fft_radix2_dit<Float, Order, 1>{}(xre, xim, wre, wim);
    }

    auto operator()(
        Float* __restrict__ xre,
        Float* __restrict__ xim,
        Float const* __restrict__ wre,
        Float const* __restrict__ wim
    ) -> void
        requires(Stage != 0 and Stage < Order)
    {
        constexpr auto const size         = 1 << Order;
        constexpr auto const stage_length = ipow<2>(Stage);
        constexpr auto const stride       = ipow<2>(Stage + 1);
        constexpr auto const tw_stride    = ipow<2>(Order - Stage - 1);

        for (auto k{0}; k < size; k += stride) {
            for (auto pair{0}; pair < stage_length; ++pair) {
                auto const twi = pair * tw_stride;
                auto const tw  = std::complex{wre[twi], wim[twi]};

                auto const i1 = k + pair;
                auto const i2 = k + pair + stage_length;

                auto const x1 = std::complex{xre[i1], xim[i1]};
                auto const x2 = std::complex{xre[i2], xim[i2]};

                auto const xn1 = x1 + tw * x2;
                xre[i1]        = xn1.real();
                xim[i1]        = xn1.imag();

                auto const xn2 = x1 - tw * x2;
                xre[i2]        = xn2.real();
                xim[i2]        = xn2.imag();
            }
        }

        split_fft_radix2_dit<Float, Order, Stage + 1>{}(xre, xim, wre, wim);
    }

    auto operator()(
        Float* __restrict__ xre,
        Float* __restrict__ xim,
        Float const* __restrict__ wre,
        Float const* __restrict__ wim
    ) -> void
        requires(Stage == Order)
    {}
};

template<typename Float, int Order>
struct static_split_fft_plan
{
    static_split_fft_plan()
    {
        auto tw = make_twiddle_lut_radix2<Float>(size_t(size()), direction::forward);
        _wfre.resize(tw.size());
        _wfim.resize(tw.size());
        _wbre.resize(tw.size());
        _wbim.resize(tw.size());
        for (auto i{0U}; i < tw.size(); ++i) {
            _wfre[i] = tw[i].real();
            _wfim[i] = tw[i].imag();
            _wbre[i] = tw[i].real();
            _wbim[i] = -tw[i].imag();
        }
    }

    [[nodiscard]] static constexpr auto size() { return 1 << Order; }

    [[nodiscard]] static constexpr auto order() { return Order; }

    auto operator()(std::span<Float> xre, std::span<Float> xim, direction dir) -> void
    {
        bitrevorder(xre, xim, _rev);

        if (dir == direction::forward) {
            split_fft_radix2_dit<Float, Order, 0>{}(xre.data(), xim.data(), _wfre.data(), _wfim.data());
        } else {
            split_fft_radix2_dit<Float, Order, 0>{}(xre.data(), xim.data(), _wbre.data(), _wbim.data());
        }
    }

private:
    std::vector<Float> _wfre;
    std::vector<Float> _wfim;
    std::vector<Float> _wbre;
    std::vector<Float> _wbim;
    std::vector<std::size_t> _rev{make_bitrevorder_table(size_t(size()))};
};

template<typename Float, int Order>
struct interleave_benchmark
{
    interleave_benchmark()
    {
        auto const signal = generate_noise_signal<std::complex<Float>>(_plan.size(), std::random_device{}());
        std::copy(signal.begin(), signal.end(), _buffer.begin());
    }

    auto operator()()
    {
        _plan(_buffer, direction::forward);
        _plan(_buffer, direction::backward);

        auto scale = [f = 1.0F / Float(1 << Order)](auto c) { return c * f; };
        std::transform(_buffer.begin(), _buffer.end(), _buffer.begin(), scale);
    }

private:
    static_fft_plan<Float, Order> _plan;
    std::vector<std::complex<Float>> _buffer{std::vector<std::complex<Float>>(_plan.size())};
};

template<typename Float, int Order>
struct split_benchmark
{
    split_benchmark()
    {
        auto const signal = generate_noise_signal<std::complex<Float>>(_plan.size(), std::random_device{}());
        std::transform(signal.begin(), signal.end(), _bufre.begin(), [](auto c) { return c.real(); });
        std::transform(signal.begin(), signal.end(), _bufim.begin(), [](auto c) { return c.imag(); });
    }

    auto operator()()
    {
        _plan(_bufre, _bufim, direction::forward);
        _plan(_bufre, _bufim, direction::backward);

        auto scale = [f = 1.0F / Float(1 << Order)](auto c) { return c * f; };
        std::transform(_bufre.begin(), _bufre.end(), _bufre.begin(), scale);
        std::transform(_bufim.begin(), _bufim.end(), _bufim.begin(), scale);
    }

private:
    static_split_fft_plan<Float, Order> _plan;
    std::vector<Float> _bufre{std::vector<Float>(_plan.size())};
    std::vector<Float> _bufim{std::vector<Float>(_plan.size())};
};

auto main() -> int
{
    static constexpr auto N = 4U;

    auto x = std::vector(N, std::complex<double>{0, 0});
    x[0]   = {1.0F, 0.0F};

    auto plan = static_fft_plan<double, bit_log2(N)>{};

    plan(x, direction::forward);
    for (auto z : x)
        std::cout << z << '\n';
    std::cout << '\n';

    plan(x, direction::backward);
    for (auto z : x)
        std::cout << z / double(N) << '\n';
    std::cout << '\n';

    timeit("static_fft_plan<float, 4>", 16, interleave_benchmark<float, 4>{});
    timeit("static_fft_plan<float, 5>", 32, interleave_benchmark<float, 5>{});
    timeit("static_fft_plan<float, 6>", 64, interleave_benchmark<float, 6>{});
    timeit("static_fft_plan<float, 7>", 128, interleave_benchmark<float, 7>{});
    timeit("static_fft_plan<float, 8>", 256, interleave_benchmark<float, 8>{});
    timeit("static_fft_plan<float, 9>", 512, interleave_benchmark<float, 9>{});
    timeit("static_fft_plan<float, 10>", 1024, interleave_benchmark<float, 10>{});
    timeit("static_fft_plan<float, 11>", 2048, interleave_benchmark<float, 11>{});
    timeit("static_fft_plan<float, 12>", 4096, interleave_benchmark<float, 12>{});
    timeit("static_fft_plan<float, 13>", 8192, interleave_benchmark<float, 13>{});
    std::cout << '\n';

    timeit("static_split_fft_plan<float, 4>", 16, split_benchmark<float, 4>{});
    timeit("static_split_fft_plan<float, 5>", 32, split_benchmark<float, 5>{});
    timeit("static_split_fft_plan<float, 6>", 64, split_benchmark<float, 6>{});
    timeit("static_split_fft_plan<float, 7>", 128, split_benchmark<float, 7>{});
    timeit("static_split_fft_plan<float, 8>", 256, split_benchmark<float, 8>{});
    timeit("static_split_fft_plan<float, 9>", 512, split_benchmark<float, 9>{});
    timeit("static_split_fft_plan<float, 10>", 1024, split_benchmark<float, 10>{});
    timeit("static_split_fft_plan<float, 11>", 2048, split_benchmark<float, 11>{});
    timeit("static_split_fft_plan<float, 12>", 4096, split_benchmark<float, 12>{});
    timeit("static_split_fft_plan<float, 13>", 8192, split_benchmark<float, 13>{});
    std::cout << '\n';

    timeit("static_fft_plan<double, 4>", 16, interleave_benchmark<double, 4>{});
    timeit("static_fft_plan<double, 5>", 32, interleave_benchmark<double, 5>{});
    timeit("static_fft_plan<double, 6>", 64, interleave_benchmark<double, 6>{});
    timeit("static_fft_plan<double, 7>", 128, interleave_benchmark<double, 7>{});
    timeit("static_fft_plan<double, 8>", 256, interleave_benchmark<double, 8>{});
    timeit("static_fft_plan<double, 9>", 512, interleave_benchmark<double, 9>{});
    timeit("static_fft_plan<double, 10>", 1024, interleave_benchmark<double, 10>{});
    timeit("static_fft_plan<double, 11>", 2048, interleave_benchmark<double, 11>{});
    timeit("static_fft_plan<double, 12>", 4096, interleave_benchmark<double, 12>{});
    timeit("static_fft_plan<double, 13>", 8192, interleave_benchmark<double, 13>{});
    std::cout << '\n';

    timeit("static_split_fft_plan<double, 4>", 16, split_benchmark<double, 4>{});
    timeit("static_split_fft_plan<double, 5>", 32, split_benchmark<double, 5>{});
    timeit("static_split_fft_plan<double, 6>", 64, split_benchmark<double, 6>{});
    timeit("static_split_fft_plan<double, 7>", 128, split_benchmark<double, 7>{});
    timeit("static_split_fft_plan<double, 8>", 256, split_benchmark<double, 8>{});
    timeit("static_split_fft_plan<double, 9>", 512, split_benchmark<double, 9>{});
    timeit("static_split_fft_plan<double, 10>", 1024, split_benchmark<double, 10>{});
    timeit("static_split_fft_plan<double, 11>", 2048, split_benchmark<double, 11>{});
    timeit("static_split_fft_plan<double, 12>", 4096, split_benchmark<double, 12>{});
    timeit("static_split_fft_plan<double, 13>", 8192, split_benchmark<double, 13>{});
    std::cout << '\n';

    return EXIT_SUCCESS;
}
