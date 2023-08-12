#include "overlap_add.hpp"

#include <neo/fft/algorithm/rms_error.hpp>
#include <neo/fft/convolution/uniform_partition.hpp>
#include <neo/fft/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <span>
#include <utility>

TEMPLATE_TEST_CASE("neo/fft/convolution: overlap_add", "", float, double)
{
    using Float = TestType;

    auto const blockSize = GENERATE(as<std::size_t>{}, 128, 256, 512);
    REQUIRE(neo::fft::next_power_of_two(blockSize * 2) == blockSize * 2);

    auto ola = neo::fft::overlap_add<Float>{blockSize, blockSize};
    REQUIRE(ola.block_size() == blockSize);
    REQUIRE(ola.filter_size() == blockSize);
    REQUIRE(ola.transform_size() == blockSize * 2UL);
    REQUIRE(ola.overlaps() == 2UL);

    auto const signal = neo::fft::generate_noise_signal<Float>(blockSize * 15UL, Catch::getSeed());

    auto output = signal;
    auto blocks = Kokkos::mdspan{output.data(), Kokkos::extents{output.size()}};

    for (auto i{0U}; i < output.size(); i += blockSize) {
        auto block = KokkosEx::submdspan(blocks, std::tuple{i, i + blockSize});
        ola(block, [](auto) {});
    }

    auto const error = neo::fft::rms_error(
        Kokkos::mdspan{signal.data(), Kokkos::extents{signal.size()}},
        Kokkos::mdspan{output.data(), Kokkos::extents{output.size()}}
    );
    REQUIRE_THAT(error, Catch::Matchers::WithinAbs(0.0, 0.00001));

    for (auto i{0ULL}; i < output.size(); ++i) {
        CAPTURE(i);
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(signal[i], 0.00001));
    }
}