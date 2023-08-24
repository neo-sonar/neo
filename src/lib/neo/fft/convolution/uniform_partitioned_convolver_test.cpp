#include "dense_convolver.hpp"
#include "sparse_convolver.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/fft/convolution/uniform_partition.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <span>

namespace {
template<typename T>
static constexpr auto is_sparse_convolver = false;

template<typename Complex, typename Overlap, typename Fdl>
static constexpr auto
    is_sparse_convolver<neo::fft::uniform_partitioned_convolver<Overlap, Fdl, neo::fft::sparse_filter<Complex>>>
    = true;

}  // namespace

static_assert(not is_sparse_convolver<neo::fft::upola_convolver<std::complex<float>>>);
static_assert(not is_sparse_convolver<neo::fft::upols_convolver<std::complex<float>>>);
static_assert(is_sparse_convolver<neo::fft::sparse_upols_convolver<std::complex<float>>>);
static_assert(is_sparse_convolver<neo::fft::sparse_upola_convolver<std::complex<float>>>);

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft/convolution: convolver",
    "",
    (neo::fft::upola_convolver,
     neo::fft::upols_convolver,
     neo::fft::sparse_upola_convolver,
     neo::fft::sparse_upols_convolver),
    (std::complex<float>, std::complex<double>)
)
{
    using Convolver = TestType;
    using Complex   = typename Convolver::value_type;
    using Float     = typename Complex::value_type;

    auto const block_size = GENERATE(as<std::size_t>{}, 128, 256, 512, 1024);

    auto const filter = neo::generate_identity_impulse<Float>(block_size, 3);
    REQUIRE(filter.extent(0) == 3);
    REQUIRE(filter.extent(1) == block_size + 1);

    auto const signal = neo::generate_noise_signal<Float>(block_size * 20UL, Catch::getSeed());
    auto output       = signal;

    auto convolver = Convolver{};
    if constexpr (is_sparse_convolver<Convolver>) {
        convolver.filter(filter.to_mdspan(), [](auto, auto, auto) { return true; });
    } else {
        convolver.filter(filter.to_mdspan());
    }

    for (auto i{0U}; i < output.size(); i += block_size) {
        auto const io_block = stdex::submdspan(output.to_mdspan(), std::tuple{i, i + block_size});
        convolver(io_block);
    }

    REQUIRE(neo::allclose(output.to_mdspan(), signal.to_mdspan()));
}
