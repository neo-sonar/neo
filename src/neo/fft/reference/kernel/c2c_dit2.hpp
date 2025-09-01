// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/bit/bit_log2.hpp>
#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/math/ipow.hpp>

#include <utility>

namespace neo::fft::kernel {

/// \ingroup neo-fft
struct c2c_dit2_v1
{
    c2c_dit2_v1() = default;

    template<inout_vector Vec>
        requires complex<typename Vec::value_type>
    auto operator()(Vec x, auto const& twiddles) const noexcept -> void
    {
        auto const size  = x.size();
        auto const order = static_cast<std::int32_t>(bit_log2(size));

        {
            // stage 0
            static constexpr auto const stage_length = 1;  // ipow<2>(0)
            static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

            for (auto k{0}; k < static_cast<int>(size); k += stride) {
                auto const i1 = k;
                auto const i2 = k + stage_length;

                auto const temp = x[i1] + x[i2];
                x[i2]           = x[i1] - x[i2];
                x[i1]           = temp;
            }
        }

        auto stage_length = 2;
        auto stride       = 4;

        for (auto stage = 1; stage < order; ++stage) {
            auto const tw_stride = ipow<2>(order - stage - 1);

            for (auto k = 0; std::cmp_less(k, size); k += stride) {
                for (auto pair = 0; pair < stage_length; ++pair) {
                    auto const tw = twiddles[pair * tw_stride];

                    auto const i1 = k + pair;
                    auto const i2 = i1 + stage_length;

                    auto const temp = x[i1] + tw * x[i2];
                    x[i2]           = x[i1] - tw * x[i2];
                    x[i1]           = temp;
                }
            }

            stage_length *= 2;
            stride *= 2;
        }
    }
};

/// \ingroup neo-fft
struct c2c_dit2_v2
{
    c2c_dit2_v2() = default;

    template<inout_vector Vec>
        requires complex<typename Vec::value_type>
    auto operator()(Vec x, auto const& twiddles) const noexcept -> void
    {
        auto const size = x.size();

        {
            // stage 0
            static constexpr auto const stage_length = 1;  // ipow<2>(0)
            static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

            for (auto k{0}; k < static_cast<int>(size); k += stride) {
                auto const i1 = k;
                auto const i2 = k + stage_length;

                auto const temp = x[i1] + x[i2];
                x[i2]           = x[i1] - x[i2];
                x[i1]           = temp;
            }
        }

        auto stage_size = 4zu;
        while (stage_size <= size) {
            auto const half_stage = stage_size / 2;
            auto const k_step     = size / stage_size;

            for (auto i{0zu}; i < size; i += stage_size) {
                for (auto k{i}; k < i + half_stage; ++k) {
                    auto const index = k - i;
                    auto const tw    = twiddles[index * k_step];

                    auto const idx1 = k;
                    auto const idx2 = k + half_stage;

                    auto const even = x[idx1];
                    auto const odd  = x[idx2];

                    auto const tmp = odd * tw;
                    x[idx1]        = even + tmp;
                    x[idx2]        = even - tmp;
                }
            }

            stage_size *= 2;
        }
    }
};

/// \ingroup neo-fft
struct c2c_dit2_v3
{
    c2c_dit2_v3() = default;

    template<inout_vector Vec>
        requires complex<typename Vec::value_type>
    auto operator()(Vec x, auto const& twiddles) const noexcept -> void
    {
        auto const size  = x.size();
        auto const order = bit_log2(size);

        {
            // stage 0
            static constexpr auto const stage_length = 1;  // ipow<2>(0)
            static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

            for (auto k{0}; k < static_cast<int>(size); k += stride) {
                auto const i1 = k;
                auto const i2 = k + stage_length;

                auto const temp = x[i1] + x[i2];
                x[i2]           = x[i1] - x[i2];
                x[i1]           = temp;
            }
        }

        for (auto stage{1zu}; stage < order; ++stage) {

            auto const stage_length = ipow<2zu>(stage);
            auto const stride       = ipow<2zu>(stage + 1zu);
            auto const tw_stride    = ipow<2zu>(order - stage - 1zu);

            for (auto k{0zu}; k < size; k += stride) {
                for (auto pair{0zu}; pair < stage_length; ++pair) {
                    auto const tw = twiddles[pair * tw_stride];

                    auto const i1 = k + pair;
                    auto const i2 = k + pair + stage_length;

                    auto const temp = x[i1] + tw * x[i2];
                    x[i2]           = x[i1] - tw * x[i2];
                    x[i1]           = temp;
                }
            }
        }
    }
};

/// \ingroup neo-fft
struct c2c_dit2_v4
{
    c2c_dit2_v4() = default;

    template<inout_vector Vec>
        requires complex<typename Vec::value_type>
    auto operator()(Vec x, auto const& twiddles) const noexcept -> void
    {
        auto const size  = x.size();
        auto const order = bit_log2(size);

        {
            // stage = 0
            static constexpr auto const stage_length = 1;  // ipow<2>(0)
            static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

            for (auto k{0}; k < static_cast<int>(size); k += stride) {
                auto const i1 = k;
                auto const i2 = k + stage_length;

                auto const temp = x[i1] + x[i2];
                x[i2]           = x[i1] - x[i2];
                x[i1]           = temp;
            }
        }

        for (auto stage{1zu}; stage < order - 1; ++stage) {

            auto const stage_length = ipow<2zu>(stage);
            auto const stride       = ipow<2zu>(stage + 1);
            auto const tw_stride    = ipow<2zu>(order - stage - 1zu);

            for (auto k{0zu}; k < size; k += stride) {
                for (auto pair{0zu}; pair < stage_length; ++pair) {
                    auto const tw = twiddles[pair * tw_stride];

                    auto const i1 = k + pair;
                    auto const i2 = k + pair + stage_length;

                    auto const temp = x[i1] + tw * x[i2];
                    x[i2]           = x[i1] - tw * x[i2];
                    x[i1]           = temp;
                }
            }
        }

        {
            // stage = order - 1
            auto const stage_length = size / 2zu;

            for (auto pair{0zu}; pair < stage_length; ++pair) {
                auto const tw = twiddles[pair];

                auto const i1 = pair;
                auto const i2 = pair + stage_length;

                auto const temp = x[i1] + tw * x[i2];
                x[i2]           = x[i1] - tw * x[i2];
                x[i1]           = temp;
            }
        }
    }
};

}  // namespace neo::fft::kernel
