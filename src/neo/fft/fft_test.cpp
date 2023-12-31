// SPDX-License-Identifier: MIT

#include "fft.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex/scalar_complex.hpp>
#include <neo/fft.hpp>
#include <neo/simd.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <random>
#include <vector>

namespace {
template<typename Complex, typename Kernel>
struct kernel_tester
{
    using plan_type = neo::fft::fallback_fft_plan<Complex, Kernel>;
};

template<typename Complex>
using kernel_v1 = kernel_tester<Complex, neo::fft::kernel::c2c_dit2_v1>;

template<typename Complex>
using kernel_v2 = kernel_tester<Complex, neo::fft::kernel::c2c_dit2_v2>;

template<typename Complex>
using kernel_v3 = kernel_tester<Complex, neo::fft::kernel::c2c_dit2_v3>;

constexpr auto execute_dit2_kernel = [](auto kernel, neo::inout_vector auto x, auto const& twiddles) -> void {
    neo::fft::bitrevorder(x);
    kernel(x, twiddles);
};

template<typename Plan>
auto test_fft_plan()
{
    using Complex = typename Plan::value_type;
    using Float   = typename Complex::value_type;

    // REQUIRE(neo::fft::next_order(2U) == 1U);
    // REQUIRE(neo::fft::next_order(3U) == 2U);

    auto const order = GENERATE(as<neo::fft::order>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    CAPTURE(order);

    auto plan = Plan{order};
    REQUIRE(plan.order() == order);
    REQUIRE(plan.size() == neo::fft::size(order));
    REQUIRE(neo::fft::next_order(plan.size()) == plan.order());

    auto const noise = neo::generate_noise_signal<Complex>(plan.size(), Catch::getSeed());

    SECTION("inplace")
    {
        auto copy = noise;
        auto io   = copy.to_mdspan();
        STATIC_REQUIRE(neo::has_default_accessor<decltype(io)>);
        STATIC_REQUIRE(neo::has_layout_left_or_right<decltype(io)>);

        neo::fft::fft(plan, io);
        neo::fft::ifft(plan, io);

        neo::scale(Float(1) / static_cast<Float>(plan.size()), io);
        REQUIRE(neo::allclose(noise.to_mdspan(), io));
    }

    SECTION("copy")
    {
        auto tmp_buf = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{noise.extents()};
        auto out_buf = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{noise.extents()};
        STATIC_REQUIRE(neo::has_default_accessor<decltype(tmp_buf.to_mdspan())>);
        STATIC_REQUIRE(neo::has_default_accessor<decltype(out_buf.to_mdspan())>);
        STATIC_REQUIRE(neo::has_layout_left_or_right<decltype(tmp_buf.to_mdspan())>);
        STATIC_REQUIRE(neo::has_layout_left_or_right<decltype(out_buf.to_mdspan())>);

        auto tmp = tmp_buf.to_mdspan();
        auto out = out_buf.to_mdspan();

        neo::fft::fft(plan, noise.to_mdspan(), tmp);
        neo::fft::ifft(plan, tmp, out);

        neo::scale(Float(1) / static_cast<Float>(plan.size()), out);
        REQUIRE(neo::allclose(noise.to_mdspan(), out));
    }

// Bug in Kokkos::layout_stride no_unique_address_emulation
#if not defined(NEO_PLATFORM_WINDOWS)
    SECTION("inplace strided")
    {
        auto buf = stdex::mdarray<Complex, stdex::dextents<size_t, 2>, stdex::layout_left>{2, plan.size()};
        auto io  = stdex::submdspan(buf.to_mdspan(), 0, stdex::full_extent);
        neo::copy(noise.to_mdspan(), io);

        STATIC_REQUIRE(neo::has_default_accessor<decltype(io)>);
        STATIC_REQUIRE_FALSE(neo::has_layout_left_or_right<decltype(io)>);

        neo::fft::fft(plan, io);
        neo::fft::ifft(plan, io);

        neo::scale(Float(1) / static_cast<Float>(plan.size()), io);
        REQUIRE(neo::allclose(noise.to_mdspan(), io));
    }
#endif
}

}  // namespace

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft: fallback_fft_plan",
    "",
    (kernel_v1, kernel_v2, kernel_v3),
    (neo::complex64, neo::complex128, std::complex<float>, std::complex<double>)
)
{
    test_fft_plan<typename TestType::plan_type>();
}

#if defined(NEO_HAS_APPLE_ACCELERATE)
TEMPLATE_TEST_CASE("neo/fft: apple_vdsp_fft_plan", "", neo::complex64, std::complex<float>, neo::complex128, std::complex<double>)
{
    test_fft_plan<neo::fft::apple_vdsp_fft_plan<TestType>>();
}
#endif

#if defined(NEO_HAS_INTEL_IPP)
TEMPLATE_TEST_CASE("neo/fft: intel_ipp_fft_plan", "", neo::complex64, std::complex<float>, neo::complex128, std::complex<double>)
{
    test_fft_plan<neo::fft::intel_ipp_fft_plan<TestType>>();
}
#endif

#if defined(NEO_HAS_INTEL_MKL)
TEMPLATE_TEST_CASE("neo/fft: intel_mkl_fft_plan", "", neo::complex64, std::complex<float>, neo::complex128, std::complex<double>)
{
    test_fft_plan<neo::fft::intel_mkl_fft_plan<TestType>>();
}
#endif

TEMPLATE_TEST_CASE("neo/fft: fft_plan", "", neo::complex64, std::complex<float>, neo::complex128, std::complex<double>)
{
    test_fft_plan<neo::fft::fft_plan<TestType>>();
}

template<typename ComplexBatch, typename Kernel>
static auto test_complex_batch_roundtrip_fft()
{
    using ScalarBatch   = typename ComplexBatch::batch_type;
    using ScalarFloat   = typename ComplexBatch::real_scalar_type;
    using ScalarComplex = std::complex<ScalarFloat>;

    auto make_noise_signal = [](auto size) {
        auto noise = neo::generate_noise_signal<ScalarComplex>(size, Catch::getSeed());
        auto buf   = stdex::mdarray<ComplexBatch, stdex::dextents<size_t, 1>>{size};
        for (auto i{0UL}; i < size; ++i) {
            buf(i) = ComplexBatch{
                ScalarBatch::broadcast(noise(i).real()),
                ScalarBatch::broadcast(noise(i).imag()),
            };
        }
        return buf;
    };

    auto make_twiddles = [](auto size, neo::fft::direction dir) {
        auto tw  = neo::fft::detail::make_radix2_twiddles<ScalarComplex>(size, dir);
        auto buf = stdex::mdarray<ComplexBatch, stdex::dextents<size_t, 1>>{tw.extents()};
        for (auto i{0UL}; i < buf.extent(0); ++i) {
            buf(i) = ComplexBatch{
                ScalarBatch::broadcast(tw(i).real()),
                ScalarBatch::broadcast(tw(i).imag()),
            };
        }
        return buf;
    };

    auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    auto const size  = 1UL << order;

    auto inout = make_noise_signal(size);

    auto const copy              = inout;
    auto const forward_twiddles  = make_twiddles(size, neo::fft::direction::forward);
    auto const backward_twiddles = make_twiddles(size, neo::fft::direction::backward);

    execute_dit2_kernel(Kernel{}, inout.to_mdspan(), forward_twiddles.to_mdspan());
    execute_dit2_kernel(Kernel{}, inout.to_mdspan(), backward_twiddles.to_mdspan());

    for (auto i{0U}; i < inout.extent(0); ++i) {
        auto const real_batch = inout(i).real();
        auto const imag_batch = inout(i).imag();

        auto reals = std::array<ScalarFloat, ScalarBatch::size>{};
        auto imags = std::array<ScalarFloat, ScalarBatch::size>{};
        real_batch.store_unaligned(reals.data());
        imag_batch.store_unaligned(imags.data());

        auto const expected_real_batch = copy(i).real();
        auto const expected_imag_batch = copy(i).imag();

        auto expected_reals = std::array<ScalarFloat, ScalarBatch::size>{};
        auto expected_imags = std::array<ScalarFloat, ScalarBatch::size>{};
        expected_real_batch.store_unaligned(expected_reals.data());
        expected_imag_batch.store_unaligned(expected_imags.data());

        auto const scalar = ScalarFloat(1) / static_cast<ScalarFloat>(size);
        neo::scale(scalar, stdex::mdspan{reals.data(), stdex::extents{reals.size()}});
        neo::scale(scalar, stdex::mdspan{imags.data(), stdex::extents{imags.size()}});

        REQUIRE(neo::allclose(
            stdex::mdspan{reals.data(), stdex::extents{reals.size()}},
            stdex::mdspan{expected_reals.data(), stdex::extents{expected_reals.size()}}
        ));

        REQUIRE(neo::allclose(
            stdex::mdspan{imags.data(), stdex::extents{imags.size()}},
            stdex::mdspan{expected_imags.data(), stdex::extents{expected_imags.size()}}
        ));
    }
}

#if defined(NEO_HAS_SIMD_SSE2)
TEMPLATE_TEST_CASE("neo/fft: radix2_kernel(simd_batch)", "", neo::pcomplex64x4, neo::pcomplex128x2)
{
    test_complex_batch_roundtrip_fft<TestType, neo::fft::kernel::c2c_dit2_v1>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::kernel::c2c_dit2_v2>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::kernel::c2c_dit2_v3>();
}
#endif

#if defined(NEO_HAS_SIMD_AVX)
TEMPLATE_TEST_CASE("neo/fft: radix2_kernel(simd_batch)", "", neo::pcomplex64x8, neo::pcomplex128x4)
{
    test_complex_batch_roundtrip_fft<TestType, neo::fft::kernel::c2c_dit2_v1>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::kernel::c2c_dit2_v2>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::kernel::c2c_dit2_v3>();
}
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
TEMPLATE_TEST_CASE("neo/fft: radix2_kernel(simd_batch)", "", neo::pcomplex64x16, neo::pcomplex128x8)
{
    test_complex_batch_roundtrip_fft<TestType, neo::fft::kernel::c2c_dit2_v1>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::kernel::c2c_dit2_v2>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::kernel::c2c_dit2_v3>();
}
#endif

#if defined(NEO_HAS_BUILTIN_FLOAT16) and defined(NEO_HAS_SIMD_F16C)
TEMPLATE_TEST_CASE("neo/fft: radix2_kernel(simd_batch)", "", neo::pcomplex32x8, neo::pcomplex32x16)
{
    using ComplexBatch  = TestType;
    using ScalarBatch   = typename ComplexBatch::batch_type;
    using ScalarFloat   = typename ComplexBatch::real_scalar_type;
    using ScalarComplex = std::complex<float>;

    auto test = [](auto kernel) {
        auto make_noise_signal = [](auto size) {
            auto noise = neo::generate_noise_signal<ScalarComplex>(size, Catch::getSeed());
            auto buf   = stdex::mdarray<ComplexBatch, stdex::dextents<size_t, 1>>{size};
            for (auto i{0UL}; i < size; ++i) {
                buf(i) = ComplexBatch{
                    ScalarBatch::broadcast(static_cast<ScalarFloat>(noise(i).real())),
                    ScalarBatch::broadcast(static_cast<ScalarFloat>(noise(i).imag())),
                };
            }
            return buf;
        };

        auto make_twiddles = [](auto size, neo::fft::direction dir) {
            auto tw  = neo::fft::detail::make_radix2_twiddles<ScalarComplex>(size, dir);
            auto buf = stdex::mdarray<ComplexBatch, stdex::dextents<size_t, 1>>{tw.extents()};
            for (auto i{0UL}; i < buf.extent(0); ++i) {
                buf(i) = ComplexBatch{
                    ScalarBatch::broadcast(static_cast<ScalarFloat>(tw(i).real())),
                    ScalarBatch::broadcast(static_cast<ScalarFloat>(tw(i).imag())),
                };
            }
            return buf;
        };

        auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6);
        auto const size  = 1UL << order;

        auto inout = make_noise_signal(size);

        auto const copy              = inout;
        auto const forward_twiddles  = make_twiddles(size, neo::fft::direction::forward);
        auto const backward_twiddles = make_twiddles(size, neo::fft::direction::backward);

        execute_dit2_kernel(kernel, inout.to_mdspan(), forward_twiddles.to_mdspan());
        execute_dit2_kernel(kernel, inout.to_mdspan(), backward_twiddles.to_mdspan());

        for (auto i{0U}; i < inout.extent(0); ++i) {
            auto const real_batch = inout(i).real();
            auto const imag_batch = inout(i).imag();

            auto reals = std::array<ScalarFloat, ScalarBatch::size>{};
            auto imags = std::array<ScalarFloat, ScalarBatch::size>{};
            real_batch.store_unaligned(reals.data());
            imag_batch.store_unaligned(imags.data());

            auto const expected_real_batch = copy(i).real();
            auto const expected_imag_batch = copy(i).imag();

            auto expected_reals = std::array<ScalarFloat, ScalarBatch::size>{};
            auto expected_imags = std::array<ScalarFloat, ScalarBatch::size>{};
            expected_real_batch.store_unaligned(expected_reals.data());
            expected_imag_batch.store_unaligned(expected_imags.data());

            auto reals_float          = std::array<float, ScalarBatch::size>{};
            auto imags_float          = std::array<float, ScalarBatch::size>{};
            auto expected_reals_float = std::array<float, ScalarBatch::size>{};
            auto expected_imags_float = std::array<float, ScalarBatch::size>{};
            std::copy(reals.begin(), reals.end(), reals_float.begin());
            std::copy(imags.begin(), imags.end(), imags_float.begin());
            std::copy(expected_reals.begin(), expected_reals.end(), expected_reals_float.begin());
            std::copy(expected_imags.begin(), expected_imags.end(), expected_imags_float.begin());

            auto const scalar = ScalarFloat(1) / static_cast<ScalarFloat>(size);
            neo::scale(scalar, stdex::mdspan{reals_float.data(), stdex::extents{reals_float.size()}});
            neo::scale(scalar, stdex::mdspan{imags_float.data(), stdex::extents{imags_float.size()}});

            REQUIRE(neo::allclose(
                stdex::mdspan{reals_float.data(), stdex::extents{reals_float.size()}},
                stdex::mdspan{expected_reals_float.data(), stdex::extents{expected_reals_float.size()}},
                0.1F
            ));

            REQUIRE(neo::allclose(
                stdex::mdspan{imags_float.data(), stdex::extents{imags_float.size()}},
                stdex::mdspan{expected_imags_float.data(), stdex::extents{expected_imags_float.size()}},
                0.1F
            ));
        }
    };

    test(neo::fft::kernel::c2c_dit2_v1{});
    test(neo::fft::kernel::c2c_dit2_v2{});
    test(neo::fft::kernel::c2c_dit2_v3{});
}
#endif
