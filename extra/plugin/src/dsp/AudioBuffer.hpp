// SPDX-License-Identifier: MIT

#pragma once

#include <neo/container/mdspan.hpp>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>

namespace neo {

template<std::floating_point Float>
struct BufferWithSampleRate
{
    juce::AudioBuffer<Float> buffer;
    double sampleRate;
};

[[nodiscard]] auto resample(BufferWithSampleRate<float> const& buf, double destSampleRate)
    -> BufferWithSampleRate<float>;

template<std::floating_point Float>
[[nodiscard]] auto to_mdarray(juce::AudioBuffer<Float> const& buffer)
    -> stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>
{
    auto result = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{
        static_cast<std::size_t>(buffer.getNumChannels()),
        static_cast<std::size_t>(buffer.getNumSamples()),
    };

    for (auto ch{0zu}; ch < result.extent(0); ++ch) {
        for (auto i{0zu}; i < result.extent(1); ++i) {
            result(ch, i) = buffer.getSample(static_cast<int>(ch), static_cast<int>(i));
        }
    }

    return result;
}

template<typename Processor>
auto processBlocks(
    Processor& processor,
    juce::dsp::AudioBlock<float const> const& input,
    juce::dsp::AudioBlock<float> const& output,
    std::size_t blockSize,
    double sampleRate
) -> void
{
    jassert(input.getNumChannels() == output.getNumChannels());
    jassert(input.getNumSamples() == output.getNumSamples());

    processor.prepare(
        juce::dsp::ProcessSpec{
            sampleRate,
            static_cast<std::uint32_t>(blockSize),
            static_cast<std::uint32_t>(input.getNumChannels()),
        }
    );

    for (auto i{0zu}; i < output.getNumSamples(); i += blockSize) {
        auto const numSamples = std::min(output.getNumSamples() - i, blockSize);

        auto in  = input.getSubBlock(i, numSamples);
        auto out = output.getSubBlock(i, numSamples);
        auto ctx = juce::dsp::ProcessContextNonReplacing<float>{in, out};
        processor.process(ctx);
    }

    processor.reset();
}

}  // namespace neo
