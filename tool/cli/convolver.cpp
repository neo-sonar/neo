#include "wav.hpp"

#include <neo/algorithm.hpp>
#include <neo/container.hpp>
#include <neo/convolution.hpp>
#include <neo/math.hpp>

#include <fmt/format.h>
#include <fmt/os.h>

#include <cstdlib>

template<neo::complex Complex>
using split_upola_convolver = neo::uniform_partitioned_convolver<
    neo::overlap_add<Complex>,
    neo::dense_split_fdl<typename Complex::value_type>,
    neo::dense_split_filter<typename Complex::value_type>>;

#if defined(NEO_HAS_SIMD_F16C) or defined(NEO_HAS_SIMD_NEON)
template<neo::complex Complex>
using split_upola_convolver_f16 = neo::uniform_partitioned_convolver<
    neo::overlap_add<Complex>,
    neo::dense_split_fdl<_Float16>,
    neo::dense_split_filter<_Float16>>;
#endif

template<typename Convolver>
[[nodiscard]] static auto
convolve(neo::audio_buffer<float> const& signal, neo::audio_buffer<float> const& impulse, int block_size = 512)
    -> neo::audio_buffer<float>
{
    auto impulse_copy = impulse;
    neo::normalize_impulse(impulse_copy.to_mdspan());
    auto const partitions = neo::uniform_partition(impulse_copy.to_mdspan(), static_cast<size_t>(block_size));

    auto output       = neo::audio_buffer<float>{signal.extent(0), signal.extent(1)};
    auto block_buffer = stdex::mdarray<float, stdex::dextents<size_t, 1>>(size_t(block_size));

    for (auto channel{0}; std::cmp_less(channel, signal.extent(0)); ++channel) {
        auto convolver  = Convolver{};
        auto const full = stdex::full_extent;
        convolver.filter(stdex::submdspan(partitions.to_mdspan(), channel, full, full));

        for (auto i{0}; std::cmp_less(i, output.extent(1)); i += block_size) {
            neo::fill(block_buffer.to_mdspan(), 0.0F);

            auto const num_samples   = std::min(static_cast<int>(output.extent(1)) - i, block_size);
            auto const input_block   = stdex::submdspan(signal.to_mdspan(), channel, std::tuple{i, i + num_samples});
            auto const process_block = stdex::submdspan(block_buffer.to_mdspan(), std::tuple{0, num_samples});
            neo::copy(input_block, process_block);

            convolver(block_buffer.to_mdspan());

            auto const output_block = stdex::submdspan(output.to_mdspan(), channel, std::tuple{i, i + num_samples});
            neo::copy(process_block, output_block);
        }
    }

    return output;
}

auto main(int argc, char** argv) -> int
{
    auto const args = std::span<char const* const>{argv, size_t(argc)};

    if (args.size() != 4) {
        fmt::println("Usage: ./neo_dsp_convolver path/to/signal.wav path/to/filter.wav path/to/output.wav");
        return EXIT_FAILURE;
    }

    auto const [signal, signal_sr]   = neo::load_wav_file<float>(args[1]);
    auto const [filter, filter_sr]   = neo::load_wav_file<float>(args[2]);
    auto const output_length         = signal.extent(1);
    auto const output_length_seconds = static_cast<double>(output_length) / signal_sr;
    auto const block_size            = 8192 * 4;

    if (signal.extent(0) != filter.extent(0)) {
        fmt::println("Channel mismatch: signal = {} filter = {}", int(signal.extent(0)), int(filter.extent(0)));
        return EXIT_FAILURE;
    }
    if (not neo::float_equality::exact(signal_sr, filter_sr)) {
        fmt::println("Sample-rate mismatch: signal = {} filter = {}", int(signal_sr), int(filter_sr));
        return EXIT_FAILURE;
    }

    fmt::println(
        "Filter: {} channels {} frames ({:.2f} sec) at {} kHz",
        static_cast<int>(filter.extent(0)),
        static_cast<int>(filter.extent(1)),
        static_cast<double>(filter.extent(1)) / filter_sr,
        static_cast<int>(filter_sr)
    );

    fmt::println(
        "Signal: {} channels {} frames ({:.2f} sec) at {} kHz",
        static_cast<int>(signal.extent(0)),
        static_cast<int>(signal.extent(1)),
        static_cast<double>(signal.extent(1)) / signal_sr,
        static_cast<int>(signal_sr)
    );

    {
        auto const start   = std::chrono::system_clock::now();
        auto output        = convolve<neo::upola_convolver<std::complex<float>>>(signal, filter, block_size);
        auto const stop    = std::chrono::system_clock::now();
        auto const runtime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);

        neo::normalize_peak(output.to_mdspan());
        neo::write_wav_file(output, signal_sr, args[3]);
        fmt::println(
            "UPOLA32: {:.2f} sec / {:.1f} x real-time",
            runtime.count(),
            output_length_seconds / runtime.count()
        );
    }

    {
        auto const start   = std::chrono::system_clock::now();
        auto output        = convolve<split_upola_convolver<std::complex<float>>>(signal, filter, block_size);
        auto const stop    = std::chrono::system_clock::now();
        auto const runtime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);

        neo::normalize_peak(output.to_mdspan());
        neo::write_wav_file(output, signal_sr, args[3]);
        fmt::println(
            "SPLIT_UPOLA32: {:.2f} sec / {:.1f} x real-time",
            runtime.count(),
            output_length_seconds / runtime.count()
        );
    }

#if defined(NEO_HAS_SIMD_F16C) or defined(NEO_HAS_SIMD_NEON)
    {
        auto const start   = std::chrono::system_clock::now();
        auto output        = convolve<split_upola_convolver_f16<std::complex<float>>>(signal, filter, block_size);
        auto const stop    = std::chrono::system_clock::now();
        auto const runtime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);

        neo::normalize_peak(output.to_mdspan());
        neo::write_wav_file(output, signal_sr, args[3]);
        fmt::println(
            "SPLIT_UPOLA16: {:.2f} sec / {:.1f} x real-time",
            runtime.count(),
            output_length_seconds / runtime.count()
        );
    }
#endif

    return EXIT_SUCCESS;
}
