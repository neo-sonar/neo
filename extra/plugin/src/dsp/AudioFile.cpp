// SPDX-License-Identifier: MIT

#include "AudioFile.hpp"

namespace neo {

auto load(juce::AudioFormatManager& formats, juce::File const& file) -> BufferWithSampleRate<float>
{
    auto reader = std::unique_ptr<juce::AudioFormatReader>{formats.createReaderFor(file.createInputStream())};
    if (reader == nullptr) {
        return {};
    }

    auto buffer = juce::AudioBuffer<float>{int(reader->numChannels), int(reader->lengthInSamples)};
    if (not reader->read(buffer.getArrayOfWritePointers(), buffer.getNumChannels(), 0, buffer.getNumSamples())) {
        return {};
    }

    return {std::move(buffer), reader->sampleRate};
}

auto loadAndResample(juce::AudioFormatManager& formats, juce::File const& file, double sampleRate)
    -> BufferWithSampleRate<float>
{
    auto result = load(formats, file);
    if (not juce::exactlyEqual(result.sampleRate, sampleRate)) {
        return resample(result, sampleRate);
    }
    return result;
}

auto writeToWavFile(
    juce::File const& file,
    stdex::mdspan<float, stdex::dextents<size_t, 2>> buffer,
    double sampleRate,
    int bitsPerSample
) -> void
{

    auto const sr   = sampleRate;
    auto const bits = bitsPerSample;
    auto const ch   = static_cast<unsigned>(buffer.extent(0));

    auto wav    = juce::WavAudioFormat{};
    auto out    = file.createOutputStream();
    auto writer = std::unique_ptr<juce::AudioFormatWriter>{wav.createWriterFor(out.get(), sr, ch, bits, {}, 0)};

    auto channels = std::vector<float const*>(ch);
    for (auto i{0zu}; i < ch; ++i) {
        channels[i] = std::addressof(buffer(i, 0));
    }

    if (writer != nullptr) {
        out.release();
        writer->writeFromFloatArrays(channels.data(), static_cast<int>(ch), static_cast<int>(buffer.extent(1)));
    }
}

auto writeToWavFile(
    juce::File const& file,
    juce::AudioBuffer<float> const& buffer,
    double sampleRate,
    int bitsPerSample
) -> void
{

    writeToWavFile(file, to_mdarray(buffer), sampleRate, bitsPerSample);
}

}  // namespace neo
