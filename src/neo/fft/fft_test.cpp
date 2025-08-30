// SPDX-License-Identifier: MIT

#include "fft.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex/scalar_complex.hpp>
#include <neo/fft.hpp>
#include <neo/simd.hpp>
#include <neo/testing/testing.hpp>

#if defined(NEO_HAS_XSIMD)
    #include <neo/config/xsimd.hpp>
#endif

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <random>
#include <vector>

namespace {
template<typename Complex, typename Kernel>
struct kernel_tester
{
    using plan_type    = neo::fft::c2c_dit2_plan<Complex, Kernel>;
    using complex_type = Complex;
    using kernel_type  = Kernel;
};

template<typename Complex>
using kernel_v1 = kernel_tester<Complex, neo::fft::kernel::c2c_dit2_v1>;

template<typename Complex>
using kernel_v2 = kernel_tester<Complex, neo::fft::kernel::c2c_dit2_v2>;

template<typename Complex>
using kernel_v3 = kernel_tester<Complex, neo::fft::kernel::c2c_dit2_v3>;

template<typename Complex>
using kernel_v4 = kernel_tester<Complex, neo::fft::kernel::c2c_dit2_v4>;

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

    SECTION("fail")
    {
        auto const next = neo::fft::next_order(Plan::max_size() + 1U);
        CAPTURE(int(next));
        REQUIRE_THROWS(Plan{neo::fft::from_order, next});
    }

    auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    CAPTURE(order);

    auto plan = Plan{neo::fft::from_order, order};
    REQUIRE(plan.order() == order);
    REQUIRE(plan.size() == neo::ipow<2zu>(order));
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

template<typename ComplexBatch, typename Kernel>
auto test_complex_batch_roundtrip_fft()
{
    using ScalarComplex = neo::value_type_t<ComplexBatch>;
    using ScalarBatch   = typename ComplexBatch::real_batch;
    using ScalarFloat   = typename ScalarComplex::value_type;

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
        auto tw  = neo::fft::make_twiddle_lut_radix2<ScalarComplex>(size, dir);
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

        REQUIRE(
            neo::allclose(
                stdex::mdspan{reals.data(), stdex::extents{reals.size()}},
                stdex::mdspan{expected_reals.data(), stdex::extents{expected_reals.size()}}
            )
        );

        REQUIRE(
            neo::allclose(
                stdex::mdspan{imags.data(), stdex::extents{imags.size()}},
                stdex::mdspan{expected_imags.data(), stdex::extents{expected_imags.size()}}
            )
        );
    }
}

}  // namespace

#if defined(NEO_HAS_APPLE_ACCELERATE)
TEMPLATE_TEST_CASE(
    "neo/fft: apple_vdsp_fft_plan",
    "",
    neo::complex64,
    std::complex<float>,
    neo::complex128,
    std::complex<double>
)
{
    test_fft_plan<neo::fft::apple_vdsp_fft_plan<TestType>>();
}
#endif

#if defined(NEO_HAS_INTEL_IPP)
TEMPLATE_TEST_CASE(
    "neo/fft: intel_ipp_fft_plan",
    "",
    neo::complex64,
    std::complex<float>,
    neo::complex128,
    std::complex<double>
)
{
    test_fft_plan<neo::fft::intel_ipp_fft_plan<TestType>>();
}
#endif

#if defined(NEO_HAS_INTEL_MKL)
TEMPLATE_TEST_CASE(
    "neo/fft: intel_mkl_fft_plan",
    "",
    neo::complex64,
    std::complex<float>,
    neo::complex128,
    std::complex<double>
)
{
    test_fft_plan<neo::fft::intel_mkl_fft_plan<TestType>>();
}
#endif

TEMPLATE_TEST_CASE("neo/fft: fft_plan", "", neo::complex64, std::complex<float>, neo::complex128, std::complex<double>)
{
    test_fft_plan<neo::fft::fft_plan<TestType>>();
}

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft: c2c_dit2_plan",
    "",
    (kernel_v1, kernel_v2, kernel_v3, kernel_v4),
    (neo::complex64, neo::complex128, std::complex<float>, std::complex<double>)
)
{
    test_fft_plan<typename TestType::plan_type>();
}

#if defined(NEO_HAS_XSIMD)
TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft: c2c_dit2_plan",
    "",
    (kernel_v1, kernel_v2, kernel_v3, kernel_v4),
    (xsimd::batch<std::complex<float>>, xsimd::batch<std::complex<double>>)
)
{
    using Complex = typename TestType::complex_type;
    using Kernel  = typename TestType::kernel_type;
    test_complex_batch_roundtrip_fft<Complex, Kernel>();
}
#endif

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft: ",
    "",
    (neo::fft::c2c_dif3_plan,
     neo::fft::c2c_dif4_plan,
     neo::fft::c2c_dif5_plan,
     neo::fft::c2c_dit4_plan,
     neo::fft::c2c_stockham_dif2r_plan,
     neo::fft::c2c_stockham_dif2i_plan,
     neo::fft::c2c_stockham_dif3_plan,
     neo::fft::c2c_stockham_dif4_plan,
     neo::fft::c2c_stockham_dif5_plan,
     neo::fft::c2c_stockham_dif8_plan,
     neo::fft::c2c_stockham_dit4_plan),
    (std::complex<float>, std::complex<double>, neo::complex64, neo::complex128)
)
{
    using namespace neo::fft;

    using Plan    = TestType;
    using Complex = typename Plan::value_type;
    using Float   = typename Complex::value_type;

    auto const order = GENERATE(range(size_t(1), static_cast<size_t>(Plan::max_order()) - 6));
    CAPTURE(order);

    auto plan        = Plan{neo::fft::from_order, order};
    auto const noise = neo::generate_noise_signal<Complex>(plan.size(), Catch::getSeed());

    SECTION("inplace")
    {
        auto copy = noise;
        auto io   = copy.to_mdspan();

        neo::fft::fft(plan, io);
        neo::fft::ifft(plan, io);

        neo::scale(Float(1) / static_cast<Float>(plan.size()), io);
        REQUIRE(neo::allclose(noise.to_mdspan(), io, Float(0.001)));
    }
}
