// SPDX-License-Identifier: MIT

#include "rfft.hpp"

#if defined(NEO_HAS_INTEL_IPP)
    #include <neo/fft/backend/ipp.hpp>
#endif

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex.hpp>
#include <neo/fft/experimental/rfft.hpp>
#include <neo/fft/fft.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <random>

namespace {

template<typename Float, typename Complex>
    requires(std::same_as<Float, typename Complex::value_type>)
struct tester
{
    using plan_type = neo::fft::fallback_rfft_plan<Float, Complex>;
};

template<typename Float>
using std_complex = tester<Float, std::complex<Float>>;

template<typename Float>
using neo_complex = tester<Float, neo::scalar_complex<Float>>;

template<typename Plan>
auto test_rfft()
{
    using Float   = typename Plan::real_type;
    using Complex = typename Plan::complex_type;

    auto const order = GENERATE(as<size_t>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);

    auto rfft = Plan{neo::fft::from_order, order};
    REQUIRE(rfft.order() == order);
    REQUIRE(rfft.size() == neo::ipow<2zu>(order));

    auto signal         = neo::generate_noise_signal<Float>(rfft.size(), Catch::getSeed());
    auto spectrum       = std::vector<Complex>(rfft.size() / 2UL + 1UL, Float(0));
    auto const original = signal;

    auto const real    = signal.to_mdspan();
    auto const complex = stdex::mdspan{spectrum.data(), stdex::extents{spectrum.size()}};
    rfft(real, complex);
    rfft(complex, real);

    neo::scale(Float(1) / static_cast<Float>(rfft.size()), real);
    REQUIRE(neo::allclose(stdex::mdspan{original.data(), stdex::extents{original.size()}}, real));
}

}  // namespace

TEMPLATE_PRODUCT_TEST_CASE("neo/fft: fallback_rfft_plan", "", (std_complex, neo_complex), (float, double))
{
    test_rfft<typename TestType::plan_type>();
}

#if defined(NEO_HAS_INTEL_IPP)
TEMPLATE_TEST_CASE("neo/fft: intel_ipp_rfft_plan", "", float, double)
{
    test_rfft<neo::fft::intel_ipp_rfft_plan<TestType>>();
}
#endif

TEMPLATE_PRODUCT_TEST_CASE("neo/fft: rfft_deinterleave", "", (std::complex, neo::scalar_complex), (float, double))
{
    using Complex = TestType;
    using Float   = typename Complex::value_type;

    auto const order      = GENERATE(as<std::size_t>{}, 4, 5, 6, 7, 8);
    auto const size       = neo::ipow<2zu>(order);
    auto const num_coeffs = size / 2 + 1;
    CAPTURE(order);
    CAPTURE(size);
    CAPTURE(num_coeffs);

    auto fft = neo::fft::fft_plan<Complex>{neo::fft::from_order, order};
    REQUIRE(fft.size() == size);
    REQUIRE(fft.order() == order);

    auto rfft = neo::fft::rfft_plan<Float, Complex>{neo::fft::from_order, order};
    REQUIRE(rfft.size() == size);
    REQUIRE(rfft.order() == order);

    auto rng    = std::mt19937{Catch::getSeed()};
    auto dist   = std::uniform_real_distribution<Float>{Float(-1), Float(1)};
    auto random = [&dist, &rng] { return dist(rng); };

    auto a = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{size};
    auto b = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{size};
    std::generate(a.data(), a.data() + a.size(), random);
    std::generate(b.data(), b.data() + b.size(), random);

    auto a_rev = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{num_coeffs};
    auto b_rev = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{num_coeffs};
    rfft(a.to_mdspan(), a_rev.to_mdspan());
    rfft(b.to_mdspan(), b_rev.to_mdspan());

    auto inout = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{size};
    auto merge = [](Float ra, Float rb) { return Complex{ra, rb}; };
    std::transform(a.data(), a.data() + a.size(), b.data(), inout.data(), merge);

    fft(inout.to_mdspan(), neo::fft::direction::forward);

    auto ca = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{num_coeffs};
    auto cb = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{num_coeffs};
    neo::fft::rfft_deinterleave(inout.to_mdspan(), ca.to_mdspan(), cb.to_mdspan());

    REQUIRE(neo::allclose(a_rev.to_mdspan(), ca.to_mdspan()));
    REQUIRE(neo::allclose(b_rev.to_mdspan(), cb.to_mdspan()));
}

TEMPLATE_TEST_CASE("neo/fft: experimental::fft", "", float, double)
{
    using Float = TestType;

    SECTION("identity")
    {
        auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8);
        auto const size  = std::size_t(1) << order;
        CAPTURE(order);
        CAPTURE(size);

        auto plan = neo::fft::experimental::c2c_dit2_plan<Float>{neo::fft::from_order, order};
        REQUIRE(plan.size() == size);
        REQUIRE(plan.order() == order);

        auto signal = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{size * 2U};
        signal(0)   = Float(1);

        // fft
        plan(signal.to_mdspan(), neo::fft::direction::forward);
        for (auto i{0U}; i < size; ++i) {
            auto const ire = i * 2U;
            auto const iim = ire + 1;

            REQUIRE(signal(ire) == Catch::Approx(1.0));
            REQUIRE(signal(iim) == Catch::Approx(0.0));
        }

        // ifft
        plan(signal.to_mdspan(), neo::fft::direction::forward);
        REQUIRE(signal(0) == Catch::Approx(1.0 * double(size)));
        REQUIRE(signal(1) == Catch::Approx(0.0));

        for (auto i{1U}; i < size; ++i) {
            auto const ire = i * 2U;
            auto const iim = ire + 1;

            REQUIRE(signal(ire) == Catch::Approx(0.0));
            REQUIRE(signal(iim) == Catch::Approx(0.0));
        }
    }

    SECTION("example")
    {
        auto const input    = std::array<Float, 8>{1, 0, 2, 0, 3, 0, 4, 0};
        auto const expected = std::array<Float, 8>{10, 0, -2, 2, -2, 0, -2, -2};

        auto x    = input;
        auto plan = neo::fft::experimental::c2c_dit2_plan<Float>{
            neo::fft::from_order,
            neo::bit_log2(input.size() / 2),
        };
        plan(stdex::mdspan{x.data(), stdex::extents{x.size()}}, neo::fft::direction::forward);

        for (auto i{0U}; i < expected.size(); ++i) {
            CAPTURE(i);
            REQUIRE(x[i] == Catch::Approx(expected[i]));
        }
    }
}

TEMPLATE_TEST_CASE("neo/fft: experimental::rfft_plan", "", float, double)
{
    using Float = TestType;

    auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8);
    auto const size  = std::size_t(1) << order;
    CAPTURE(order);
    CAPTURE(size);

    auto plan = neo::fft::experimental::rfft_plan<Float>{neo::fft::from_order, order};
    REQUIRE(plan.order() == order);
    REQUIRE(plan.size() == size);

    SECTION("identity")
    {
        auto buffer_storage = stdex::mdarray<Float, stdex::dextents<size_t, 1>>(size);

        auto buffer = buffer_storage.to_mdspan();
        buffer[0]   = Float(1);

        plan(buffer, neo::fft::direction::forward);
        REQUIRE(buffer[0] == Catch::Approx(1.0));
        REQUIRE(buffer[1] == Catch::Approx(1.0));
        for (auto i{2U}; i < buffer.extent(0); i += 2) {
            CAPTURE(i);
            REQUIRE(buffer[i] == Catch::Approx(1.0));
            REQUIRE(buffer[i + 1] == Catch::Approx(0.0));
        }

        plan(buffer, neo::fft::direction::backward);
        REQUIRE(buffer[0] == Catch::Approx(1.0 * double(size) / 2.0));
    }

    SECTION("random")
    {
        auto const signal = neo::generate_noise_signal<Float>(size, Catch::getSeed());
        auto copy         = signal;

        plan(copy.to_mdspan(), neo::fft::direction::forward);
        plan(copy.to_mdspan(), neo::fft::direction::backward);

        for (auto i{0U}; i < signal.extent(0); ++i) {
            CAPTURE(i);
            REQUIRE(copy(i) == Catch::Approx(signal(i) * double(size) / 2.0).scale(double(order)));
        }
    }
}
