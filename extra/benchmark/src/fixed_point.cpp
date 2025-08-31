// SPDX-License-Identifier: MIT

#include <neo/config.hpp>

#include <neo/algorithm.hpp>
#include <neo/container.hpp>
#include <neo/fixed_point.hpp>

#include <neo/testing/benchmark.hpp>
#include <neo/testing/testing.hpp>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {
template<typename Func>
auto timeit(std::string_view name, size_t size_of_t, size_t n, Func func)
{
    using microseconds = std::chrono::duration<double, std::micro>;

    auto const size       = n;
    auto const iterations = 50'000zu;
    auto const margin     = iterations / 20zu;

    auto all_runs = std::vector<double>(iterations);

    func();
    func();
    func();

    for (auto i{0zu}; i < iterations; ++i) {
        auto start = std::chrono::system_clock::now();
        func();
        auto stop = std::chrono::system_clock::now();

        all_runs[i] = std::chrono::duration_cast<microseconds>(stop - start).count();
    }

    auto const runs = std::span{all_runs}.subspan(margin, all_runs.size() - static_cast<std::size_t>(margin) * 2);
    auto const avg  = std::reduce(runs.begin(), runs.end(), 0.0) / double(runs.size());
    auto const items_per_sec      = static_cast<int>(std::lround(double(size) / avg));
    auto const mega_bytes_per_sec = std::round(double(size * size_of_t) / avg) / 1000.0;
    std::printf("%-32s avg: %.2fus - GB/sec: %.2f - N/usec: %d\n", name.data(), avg, mega_bytes_per_sec, items_per_sec);
}

template<typename FloatOrComplex, std::size_t Size>
struct float_mul
{
    explicit float_mul() : _lhs(Size), _rhs(Size), _out(Size)
    {
#if defined(NEO_HAS_BUILTIN_FLOAT16)
        if constexpr (std::same_as<FloatOrComplex, _Float16>) {
            auto const lhs = neo::generate_noise_signal<float>(Size, std::random_device{}());
            auto const rhs = neo::generate_noise_signal<float>(Size, std::random_device{}());
            neo::copy(lhs.to_mdspan(), _lhs.to_mdspan());
            neo::copy(rhs.to_mdspan(), _rhs.to_mdspan());
            return;
        }
#endif

        _lhs = neo::generate_noise_signal<FloatOrComplex>(Size, std::random_device{}());
        _rhs = neo::generate_noise_signal<FloatOrComplex>(Size, std::random_device{}());
    }

    auto operator()() noexcept -> void
    {
        neo::multiply(_lhs.to_mdspan(), _rhs.to_mdspan(), _out.to_mdspan());
        neo::do_not_optimize(_out(0));
    }

private:
    stdex::mdarray<FloatOrComplex, stdex::dextents<size_t, 1>> _lhs;
    stdex::mdarray<FloatOrComplex, stdex::dextents<size_t, 1>> _rhs;
    stdex::mdarray<FloatOrComplex, stdex::dextents<size_t, 1>> _out;
};

template<typename FloatOrFixed, std::size_t Size>
struct cmulp
{
    explicit cmulp() : _lhs(2, Size), _rhs(2, Size), _out(2, Size)
    {
        if constexpr (std::same_as<FloatOrFixed, neo::q7> or std::same_as<FloatOrFixed, neo::q15>
                      or std::same_as<FloatOrFixed, neo::fixed_point<int16_t, 14>>) {
            auto copy_to_fixed_point = [](auto src, auto dest) {
                for (auto i{0}; i < src.extent(0); ++i) {
                    dest[i] = FloatOrFixed(src[i]);
                }
            };
            auto lr = neo::generate_noise_signal<float>(Size, std::random_device{}());
            auto li = neo::generate_noise_signal<float>(Size, std::random_device{}());
            auto rr = neo::generate_noise_signal<float>(Size, std::random_device{}());
            auto ri = neo::generate_noise_signal<float>(Size, std::random_device{}());
            copy_to_fixed_point(lr.to_mdspan(), stdex::submdspan(_lhs.to_mdspan(), 0, stdex::full_extent));
            copy_to_fixed_point(li.to_mdspan(), stdex::submdspan(_lhs.to_mdspan(), 1, stdex::full_extent));
            copy_to_fixed_point(rr.to_mdspan(), stdex::submdspan(_rhs.to_mdspan(), 0, stdex::full_extent));
            copy_to_fixed_point(ri.to_mdspan(), stdex::submdspan(_rhs.to_mdspan(), 1, stdex::full_extent));
        } else {
            auto lr = neo::generate_noise_signal<FloatOrFixed>(Size, std::random_device{}());
            auto li = neo::generate_noise_signal<FloatOrFixed>(Size, std::random_device{}());
            auto rr = neo::generate_noise_signal<FloatOrFixed>(Size, std::random_device{}());
            auto ri = neo::generate_noise_signal<FloatOrFixed>(Size, std::random_device{}());
            neo::copy(lr.to_mdspan(), stdex::submdspan(_lhs.to_mdspan(), 0, stdex::full_extent));
            neo::copy(li.to_mdspan(), stdex::submdspan(_lhs.to_mdspan(), 1, stdex::full_extent));
            neo::copy(rr.to_mdspan(), stdex::submdspan(_rhs.to_mdspan(), 0, stdex::full_extent));
            neo::copy(ri.to_mdspan(), stdex::submdspan(_rhs.to_mdspan(), 1, stdex::full_extent));
        }
    }

    auto operator()() noexcept -> void
    {
        auto const lhs_real = stdex::submdspan(_lhs.to_mdspan(), 0, stdex::full_extent);
        auto const lhs_imag = stdex::submdspan(_lhs.to_mdspan(), 1, stdex::full_extent);

        auto const rhs_real = stdex::submdspan(_rhs.to_mdspan(), 0, stdex::full_extent);
        auto const rhs_imag = stdex::submdspan(_rhs.to_mdspan(), 1, stdex::full_extent);

        auto const out_real = stdex::submdspan(_out.to_mdspan(), 0, stdex::full_extent);
        auto const out_imag = stdex::submdspan(_out.to_mdspan(), 1, stdex::full_extent);

        auto const* NEO_RESTRICT lre = lhs_real.data_handle();
        auto const* NEO_RESTRICT lim = lhs_imag.data_handle();
        auto const* NEO_RESTRICT rre = rhs_real.data_handle();
        auto const* NEO_RESTRICT rim = rhs_imag.data_handle();
        auto* NEO_RESTRICT out_re    = out_real.data_handle();
        auto* NEO_RESTRICT out_im    = out_imag.data_handle();

        for (auto i{0}; std::cmp_less(i, out_real.extent(0)); ++i) {
            out_re[i] = lre[i] * rre[i] - lim[i] * rim[i];
            out_im[i] = lre[i] * rim[i] + lim[i] * rre[i];
        }

        neo::do_not_optimize(out_real[0]);
        neo::do_not_optimize(out_imag[0]);
    }

private:
    stdex::mdarray<FloatOrFixed, stdex::dextents<size_t, 2>> _lhs;
    stdex::mdarray<FloatOrFixed, stdex::dextents<size_t, 2>> _rhs;
    stdex::mdarray<FloatOrFixed, stdex::dextents<size_t, 2>> _out;
};

template<typename FixedBatch, std::size_t Size>
struct cmulp_batch_fixed_point
{
    explicit cmulp_batch_fixed_point()
        : _lhs(2, Size / FixedBatch::size)
        , _rhs(2, Size / FixedBatch::size)
        , _out(2, Size / FixedBatch::size)
    {
        auto copy_to_fixed_point = [](auto src, auto dest) {
            for (auto i{0}; i < src.extent(0); ++i) {
                auto const fp = typename FixedBatch::value_type{src[i]};
                dest[i]       = FixedBatch::broadcast(fp);
            }
        };
        auto lr = neo::generate_noise_signal<float>(Size / FixedBatch::size, std::random_device{}());
        auto li = neo::generate_noise_signal<float>(Size / FixedBatch::size, std::random_device{}());
        auto rr = neo::generate_noise_signal<float>(Size / FixedBatch::size, std::random_device{}());
        auto ri = neo::generate_noise_signal<float>(Size / FixedBatch::size, std::random_device{}());
        copy_to_fixed_point(lr.to_mdspan(), stdex::submdspan(_lhs.to_mdspan(), 0, stdex::full_extent));
        copy_to_fixed_point(li.to_mdspan(), stdex::submdspan(_lhs.to_mdspan(), 1, stdex::full_extent));
        copy_to_fixed_point(rr.to_mdspan(), stdex::submdspan(_rhs.to_mdspan(), 0, stdex::full_extent));
        copy_to_fixed_point(ri.to_mdspan(), stdex::submdspan(_rhs.to_mdspan(), 1, stdex::full_extent));
    }

    auto operator()() noexcept -> void
    {
        auto const lhs_real = stdex::submdspan(_lhs.to_mdspan(), 0, stdex::full_extent);
        auto const lhs_imag = stdex::submdspan(_lhs.to_mdspan(), 1, stdex::full_extent);

        auto const rhs_real = stdex::submdspan(_rhs.to_mdspan(), 0, stdex::full_extent);
        auto const rhs_imag = stdex::submdspan(_rhs.to_mdspan(), 1, stdex::full_extent);

        auto const out_real = stdex::submdspan(_out.to_mdspan(), 0, stdex::full_extent);
        auto const out_imag = stdex::submdspan(_out.to_mdspan(), 1, stdex::full_extent);

        for (auto i{0}; std::cmp_less(i, out_real.extent(0)); ++i) {
            out_real[i] = lhs_real[i] * rhs_real[i] - lhs_imag[i] * rhs_imag[i];
            out_imag[i] = lhs_real[i] * rhs_imag[i] + lhs_imag[i] * rhs_real[i];
        }

        neo::do_not_optimize(out_real[0]);
        neo::do_not_optimize(out_imag[0]);
    }

private:
    stdex::mdarray<FixedBatch, stdex::dextents<size_t, 2>> _lhs;
    stdex::mdarray<FixedBatch, stdex::dextents<size_t, 2>> _rhs;
    stdex::mdarray<FixedBatch, stdex::dextents<size_t, 2>> _out;
};

template<typename FloatBatch, std::size_t Size>
struct cmulp_batch_float
{
    explicit cmulp_batch_float()
        : _lhs(2, Size / FloatBatch::size)
        , _rhs(2, Size / FloatBatch::size)
        , _out(2, Size / FloatBatch::size)
    {
        auto copy_to_fixed_point = [](auto src, auto dest) {
            for (auto i{0}; i < src.extent(0); ++i) {
                dest[i] = FloatBatch::broadcast(src[i]);
            }
        };
        auto lr = neo::generate_noise_signal<float>(Size / FloatBatch::size, std::random_device{}());
        auto li = neo::generate_noise_signal<float>(Size / FloatBatch::size, std::random_device{}());
        auto rr = neo::generate_noise_signal<float>(Size / FloatBatch::size, std::random_device{}());
        auto ri = neo::generate_noise_signal<float>(Size / FloatBatch::size, std::random_device{}());
        copy_to_fixed_point(lr.to_mdspan(), stdex::submdspan(_lhs.to_mdspan(), 0, stdex::full_extent));
        copy_to_fixed_point(li.to_mdspan(), stdex::submdspan(_lhs.to_mdspan(), 1, stdex::full_extent));
        copy_to_fixed_point(rr.to_mdspan(), stdex::submdspan(_rhs.to_mdspan(), 0, stdex::full_extent));
        copy_to_fixed_point(ri.to_mdspan(), stdex::submdspan(_rhs.to_mdspan(), 1, stdex::full_extent));
    }

    auto operator()() noexcept -> void
    {
        auto const lhs_real = stdex::submdspan(_lhs.to_mdspan(), 0, stdex::full_extent);
        auto const lhs_imag = stdex::submdspan(_lhs.to_mdspan(), 1, stdex::full_extent);

        auto const rhs_real = stdex::submdspan(_rhs.to_mdspan(), 0, stdex::full_extent);
        auto const rhs_imag = stdex::submdspan(_rhs.to_mdspan(), 1, stdex::full_extent);

        auto const out_real = stdex::submdspan(_out.to_mdspan(), 0, stdex::full_extent);
        auto const out_imag = stdex::submdspan(_out.to_mdspan(), 1, stdex::full_extent);

        for (auto i{0}; std::cmp_less(i, out_real.extent(0)); ++i) {
            out_real[i] = lhs_real[i] * rhs_real[i] - lhs_imag[i] * rhs_imag[i];
            out_imag[i] = lhs_real[i] * rhs_imag[i] + lhs_imag[i] * rhs_real[i];
        }

        neo::do_not_optimize(out_real[0]);
        neo::do_not_optimize(out_imag[0]);
    }

private:
    stdex::mdarray<FloatBatch, stdex::dextents<size_t, 2>> _lhs;
    stdex::mdarray<FloatBatch, stdex::dextents<size_t, 2>> _rhs;
    stdex::mdarray<FloatBatch, stdex::dextents<size_t, 2>> _out;
};

template<typename Float, typename Int>
auto compress_float(auto val)
{
    static constexpr auto scale = static_cast<Float>(std::numeric_limits<Int>::max());
    return static_cast<Int>(std::lround(val * scale));
}

template<typename FloatOrComplex, typename IntOrComplex>
auto compress_complex(auto val)
{
    static_assert(neo::complex<IntOrComplex>);

    using Float                 = typename FloatOrComplex::value_type;
    using Int                   = typename IntOrComplex::value_type;
    static constexpr auto scale = static_cast<Float>(std::numeric_limits<Int>::max());

    return IntOrComplex(
        static_cast<Int>(std::lround(val.real() * scale)),
        static_cast<Int>(std::lround(val.imag() * scale))
    );
}

template<typename FloatOrComplex, typename IntOrComplex, std::size_t Size>
struct cfloat_mul
{
    explicit cfloat_mul() : _lhs(Size), _rhs(Size), _out(Size)
    {
        auto noise_a  = neo::generate_noise_signal<FloatOrComplex>(Size, std::random_device{}());
        auto noise_b  = neo::generate_noise_signal<FloatOrComplex>(Size, std::random_device{}());
        auto compress = [](FloatOrComplex val) {
            if constexpr (neo::complex<FloatOrComplex>) {
                return compress_complex<FloatOrComplex, IntOrComplex>(val);
            } else {
                return compress_float<FloatOrComplex, IntOrComplex>(val);
            }
        };

        std::transform(noise_a.data(), noise_a.data() + noise_a.size(), _lhs.data(), compress);
        std::transform(noise_b.data(), noise_b.data() + noise_b.size(), _rhs.data(), compress);
    }

    auto operator()() -> void
    {
        using nested_accessor = typename stdex::mdspan<IntOrComplex, stdex::dextents<size_t, 1>>::accessor_type;
        using real_accessor   = neo::compressed_accessor<FloatOrComplex, nested_accessor>;
        using extents         = stdex::dextents<size_t, 1>;
        using real_mdspan     = stdex::mdspan<FloatOrComplex, extents, stdex::layout_right, real_accessor>;

        neo::multiply(real_mdspan(_lhs.to_mdspan()), real_mdspan(_rhs.to_mdspan()), _out.to_mdspan());
    }

private:
    stdex::mdarray<IntOrComplex, stdex::dextents<size_t, 1>> _lhs;
    stdex::mdarray<IntOrComplex, stdex::dextents<size_t, 1>> _rhs;
    stdex::mdarray<FloatOrComplex, stdex::dextents<size_t, 1>> _out;
};

template<typename FixedPoint, std::size_t Size>
struct fixed_point_mul
{
    explicit fixed_point_mul() : _lhs(Size), _rhs(Size), _out(Size)
    {
        auto rng  = std::mt19937{std::random_device{}()};
        auto dist = std::uniform_real_distribution<float>{-1.0F, 1.0F};
        auto gen  = [&] {
            if constexpr (neo::complex<FixedPoint>) {
                using FP = typename FixedPoint::value_type;
                return FixedPoint{FP{dist(rng)}, FP{dist(rng)}};
            } else {
                return FixedPoint{dist(rng)};
            }
        };
        std::generate(_lhs.begin(), _lhs.end(), gen);
        std::generate(_rhs.begin(), _rhs.end(), gen);
    }

    auto operator()() -> void
    {
        if constexpr (neo::complex<FixedPoint>) {
            auto const left   = stdex::mdspan{_lhs.data(), stdex::extents{Size}};
            auto const right  = stdex::mdspan{_rhs.data(), stdex::extents{Size}};
            auto const output = stdex::mdspan{_out.data(), stdex::extents{Size}};
            neo::multiply(left, right, output);
        } else {
            auto const left   = std::span{std::as_const(_lhs)};
            auto const right  = std::span{std::as_const(_rhs)};
            auto const output = std::span{_out};
            neo::multiply(left, right, output);
        }
    }

private:
    std::vector<FixedPoint> _lhs;
    std::vector<FixedPoint> _rhs;
    std::vector<FixedPoint> _out;
};

#if defined(NEO_HAS_BUILTIN_FLOAT16) and defined(NEO_HAS_ISA_F16C)
template<std::size_t Size>
struct float16_mul_bench
{
    explicit float16_mul_bench() : _lhs(Size), _rhs(Size), _out(Size)
    {
        auto rng  = std::mt19937{std::random_device{}()};
        auto dist = std::uniform_real_distribution<float>{-1.0F, 1.0F};
        std::generate(_lhs.begin(), _lhs.end(), [&] { return _cvtss_sh(dist(rng), _MM_FROUND_CUR_DIRECTION); });
        std::generate(_rhs.begin(), _rhs.end(), [&] { return _cvtss_sh(dist(rng), _MM_FROUND_CUR_DIRECTION); });
    }

    auto operator()() -> void
    {
        static constexpr auto const vectorSize = 128 / 16;
        static_assert((Size % vectorSize) == 0);

        for (auto i{0}; i < Size; i += vectorSize) {
            auto const lhsWord = _mm_loadu_si128(reinterpret_cast<__m128i const*>(_lhs.data() + i));
            auto const rhsWord = _mm_loadu_si128(reinterpret_cast<__m128i const*>(_rhs.data() + i));

            auto const lhs     = _mm256_cvtph_ps(lhsWord);
            auto const rhs     = _mm256_cvtph_ps(rhsWord);
            auto const product = _mm256_mul_ps(lhs, rhs);

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(_out.data() + i),
                _mm256_cvtps_ph(product, _MM_FROUND_CUR_DIRECTION)
            );
        }
    }

private:
    std::vector<std::uint16_t> _lhs;
    std::vector<std::uint16_t> _rhs;
    std::vector<std::uint16_t> _out;
};
#endif

}  // namespace

auto main() -> int
{
    static constexpr auto n = 131072zu;

    timeit("mul(q7):     ", 1, n, fixed_point_mul<neo::q7, n>{});
    timeit("mul(q15):    ", 2, n, fixed_point_mul<neo::q15, n>{});
    timeit("mul(fxp_14): ", 2, n, fixed_point_mul<neo::fixed_point<int16_t, 14>, n>{});
#if defined(NEO_HAS_BUILTIN_FLOAT16) and defined(NEO_HAS_ISA_F16C)
    timeit("mul(f16f32): ", 2, n, float16_mul_bench<n>{});
#endif
#if defined(NEO_HAS_ISA_F16C) or defined(NEO_HAS_ISA_NEON)
    // timeit("mul(_Float16): ", 2, n, float_mul<_Float16, n>{});
#endif

    timeit("mul(float):    ", 4, n, float_mul<float, n>{});
    timeit("mul(double):   ", 8, n, float_mul<double, n>{});
    timeit("mul(cf8):      ", 1, n, cfloat_mul<float, int8_t, n>{});
    timeit("mul(cf16):     ", 2, n, cfloat_mul<float, int16_t, n>{});
    std::puts("\n");

    // timeit("cmul(complex<cf8>):         ", 2, n, cfloat_mul<neo::complex64, neo::scalar_complex<int8_t>, n>{});
    // timeit("cmul(complex<cf16>):        ", 4, n, cfloat_mul<neo::complex64, neo::scalar_complex<int16_t>, n>{});

#if defined(NEO_HAS_ISA_F16C) or defined(NEO_HAS_ISA_NEON)
    // timeit("cmul(complex32):            ", 4, n, float_mul<neo::scalar_complex<_Float16>, n>{});
#endif

    timeit("cmul(complex64):            ", 8, n, float_mul<neo::complex64, n>{});
    timeit("cmul(complex128):           ", 16, n, float_mul<neo::complex128, n>{});
    timeit("cmul(std::complex<float>):  ", 8, n, float_mul<std::complex<float>, n>{});
    timeit("cmul(std::complex<double>): ", 16, n, float_mul<std::complex<double>, n>{});
    std::puts("\n");

    // timeit("cmulp(q7):       ", 2, n, cmulp<neo::q7, n>{});
    // timeit("cmulp(q15):      ", 4, n, cmulp<neo::q15, n>{});
    // timeit("cmulp(fxp_14):   ", 4, n, cmulp<neo::fixed_point<int16_t, 14>, n>{});

#if defined(NEO_HAS_ISA_F16C) or defined(NEO_HAS_ISA_NEON)
    // timeit("cmulp(_Float16): ", 4, n, cmulp<_Float16, n>{});
#endif
    timeit("cmulp(float):    ", 8, n, cmulp<float, n>{});
    timeit("cmulp(double):   ", 16, n, cmulp<double, n>{});
    std::puts("\n");

#if defined(NEO_HAS_ISA_SSE41)
    timeit("cmulp_batch_fixed_point(q7x16):  ", 2, n, cmulp_batch_fixed_point<neo::q7x16, n>{});
    timeit("cmulp_batch_fixed_point(q15x8):  ", 4, n, cmulp_batch_fixed_point<neo::q15x8, n>{});

#endif

#if defined(NEO_HAS_ISA_AVX2)
    timeit("cmulp_batch_fixed_point(q15x16): ", 4, n, cmulp_batch_fixed_point<neo::q15x16, n>{});
#endif

#if defined(NEO_HAS_ISA_AVX512BW)
    timeit("cmulp_batch_fixed_point(q15x32): ", 4, n, cmulp_batch_fixed_point<neo::q15x32, n>{});
#endif
    std::puts("\n");

#if defined(NEO_HAS_ISA_SSE2)
    timeit("cmulp_batch_float(float32x4): ", 8, n, cmulp_batch_float<neo::float32x4, n>{});
    timeit("cmulp_batch_float(float64x2): ", 16, n, cmulp_batch_float<neo::float64x2, n>{});
#endif

#if defined(NEO_HAS_ISA_AVX)
    timeit("cmulp_batch_float(float32x8): ", 8, n, cmulp_batch_float<neo::float32x8, n>{});
    timeit("cmulp_batch_float(float64x4): ", 16, n, cmulp_batch_float<neo::float64x4, n>{});
#endif

#if defined(NEO_HAS_ISA_AVX512F)
    timeit("cmulp_batch_float(float32x16): ", 8, n, cmulp_batch_float<neo::float32x16, n>{});
    timeit("cmulp_batch_float(float64x8): ", 16, n, cmulp_batch_float<neo::float64x8, n>{});
#endif

    return EXIT_SUCCESS;
}
