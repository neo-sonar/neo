// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/container/mdspan.hpp>

#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wcast-align"
    #pragma clang diagnostic ignored "-Wimplicit-int-conversion"
    #pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
    #pragma clang diagnostic ignored "-Wshift-sign-overflow"
    #pragma clang diagnostic ignored "-Wsign-conversion"
    #pragma clang diagnostic ignored "-Wswitch-enum"
    #pragma clang diagnostic ignored "-Wunused-but-set-variable"
    #pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wcast-align"
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wswitch-enum"
    #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
    #pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif

#include "dr_wav.h"

#if defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#include <fmt/format.h>
#include <fmt/os.h>

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace neo {

template<std::floating_point Float>
using audio_buffer = stdex::mdarray<Float, stdex::dextents<size_t, 2>, stdex::layout_left>;

template<std::floating_point Float>
[[nodiscard]] auto load_wav_file(std::filesystem::path const& path) -> std::pair<audio_buffer<Float>, double>
{
    auto path_str = path.string();
    auto wav_file = drwav{};
    if (not drwav_init_file(&wav_file, path_str.c_str(), nullptr)) {
        fmt::println("Could not open wav-file at: {}", path_str.c_str());
        return {};
    }

    auto interleaved = audio_buffer<float>{wav_file.channels, wav_file.totalPCMFrameCount};
    auto const read  = drwav_read_pcm_frames_f32(&wav_file, wav_file.totalPCMFrameCount, interleaved.data());
    if (read != interleaved.extent(1)) {
        fmt::println("Frames read size mismatch, expected: {} actual: {}", int(interleaved.extent(1)), int(read));
        drwav_uninit(&wav_file);
        return {};
    }

    drwav_uninit(&wav_file);

    if constexpr (std::same_as<Float, float>) {
        return {
            std::move(interleaved),
            static_cast<double>(wav_file.sampleRate),
        };
    } else {
        auto converted = audio_buffer<Float>{
            wav_file.channels,
            wav_file.totalPCMFrameCount,
        };
        copy(interleaved, converted);

        return {
            std::move(converted),
            static_cast<double>(wav_file.sampleRate),
        };
    }
}

inline auto
write_wav_file(audio_buffer<float> const& buffer, double sampleRate, std::filesystem::path const& path) -> void
{
    auto format          = drwav_data_format{};
    format.container     = drwav_container_riff;
    format.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels      = static_cast<std::uint32_t>(buffer.extent(0));
    format.sampleRate    = static_cast<std::uint32_t>(sampleRate);
    format.bitsPerSample = 32;

    auto path_str = path.string();
    auto wav_file = drwav{};
    drwav_init_file_write(&wav_file, path_str.c_str(), &format, nullptr);
    drwav_write_pcm_frames(&wav_file, buffer.extent(1), buffer.data());
    drwav_uninit(&wav_file);
}
}  // namespace neo
