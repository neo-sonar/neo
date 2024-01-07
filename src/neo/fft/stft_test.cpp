// SPDX-License-Identifier: MIT

#include "stft.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEST_CASE("neo/fft: detail::num_sftf_frames")
{
    REQUIRE(neo::fft::detail::num_sftf_frames(1024, 128, 0) == 8);
    REQUIRE(neo::fft::detail::num_sftf_frames(1024, 256, 0) == 4);
    REQUIRE(neo::fft::detail::num_sftf_frames(1024, 256, 128) == 8);
}

TEMPLATE_TEST_CASE("neo/fft: stft", "", float, double)
{
    using Float = TestType;

    auto const signal_length = GENERATE(as<std::size_t>{}, 2040, 2048);
    auto const signal        = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{1, signal_length};

    auto const no_overlap_options = neo::fft::stft_options<Float>{
        .frame_size     = 256,
        .transform_size = 256,
        .overlap_size   = 0,
    };

    auto no_overlap = neo::fft::stft(signal.to_mdspan(), no_overlap_options);
    REQUIRE(no_overlap.extent(0) == 1);
    REQUIRE(no_overlap.extent(1) == 8);
    REQUIRE(no_overlap.extent(2) == 129);

    auto half_overlap = neo::fft::stft(signal.to_mdspan(), 256);
    REQUIRE(half_overlap.extent(0) == 1);
    REQUIRE(half_overlap.extent(1) == 16);
    REQUIRE(half_overlap.extent(2) == 129);
}
