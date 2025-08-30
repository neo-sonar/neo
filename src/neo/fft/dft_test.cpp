// SPDX-License-Identifier: MIT

#include "dft.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

namespace {

template<typename Plan>
auto test_dft_plan()
{
    using Complex = typename Plan::value_type;
    using Float   = typename Complex::value_type;

    auto const max_size = sizeof(Float) == 8 ? 512 : 128;
    auto const size     = GENERATE(range(std::size_t(2), std::size_t(max_size)));
    CAPTURE(size);

    auto plan = Plan{size};
    REQUIRE(plan.size() == size);

    SECTION("identity")
    {
        auto x_buf = stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>>{size};
        auto x     = x_buf.to_mdspan();
        x[0]       = Float(1);

        neo::fft::dft(plan, x);

        for (auto i{0U}; i < size; ++i) {
            CAPTURE(i);
            REQUIRE(x[i].real() == Catch::Approx(Float(1)));
        }

        plan(x, neo::fft::direction::backward);
        REQUIRE(x[0].real() == Catch::Approx(Float(1) * Float(size)));
    }

    SECTION("random")
    {
        auto const signal = neo::generate_noise_signal<Complex>(size, Catch::getSeed());
        auto const sig    = signal.to_mdspan();

        auto x_buf = signal;
        auto x     = x_buf.to_mdspan();

        neo::fft::dft(plan, x);
        neo::fft::idft(plan, x);

        for (auto i{0U}; i < x.extent(0); ++i) {
            CAPTURE(i);

            auto const n     = static_cast<double>(size);
            auto const scale = neo::bit_log2(size) * 4;
            REQUIRE(x[i].real() == Catch::Approx(double(sig[i].real()) * n).scale(scale));
            REQUIRE(x[i].imag() == Catch::Approx(double(sig[i].imag()) * n).scale(scale));
        }
    }
}
}  // namespace

TEMPLATE_TEST_CASE("neo/fft: dft", "", float, double)
{
    using Float   = TestType;
    using Complex = std::complex<Float>;

    auto const size = GENERATE(as<std::size_t>{}, 4, 16, 21);

    auto const original = neo::generate_noise_signal<Complex>(size, Catch::getSeed());

    auto in_buf  = original;
    auto out_buf = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{in_buf.size()};

    auto const in  = in_buf.to_mdspan();
    auto const out = out_buf.to_mdspan();

    neo::fft::dft(in, out, neo::fft::direction::forward);
    neo::fft::dft(out, in, neo::fft::direction::backward);

    neo::scale(Float(1) / static_cast<Float>(size), in);
    REQUIRE(neo::allclose(original.to_mdspan(), in));
}

TEMPLATE_TEST_CASE("neo/fft: dft_plan", "", std::complex<float>, std::complex<double>)
{
    test_dft_plan<neo::fft::dft_plan<TestType>>();
}

TEMPLATE_TEST_CASE("neo/fft: fallback_dft_plan", "", std::complex<float>, std::complex<double>)
{
    test_dft_plan<neo::fft::fallback_dft_plan<TestType>>();
}

#if defined(NEO_HAS_INTEL_IPP)
TEMPLATE_TEST_CASE("neo/fft: intel_ipp_dft_plan", "", std::complex<float>, std::complex<double>)
{
    test_dft_plan<neo::fft::intel_ipp_dft_plan<TestType>>();
}
#endif
