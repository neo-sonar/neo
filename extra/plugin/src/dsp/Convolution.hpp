#pragma once

#include "dsp/AudioFile.hpp"

#include <neo/convolution.hpp>

#include <juce_dsp/juce_dsp.h>

namespace neo {

struct Convolution
{
    using Latency    = juce::dsp::Convolution::Latency;
    using NonUniform = juce::dsp::Convolution::NonUniform;

    using Normalise = juce::dsp::Convolution::Normalise;
    using Trim      = juce::dsp::Convolution::Trim;
    using Stereo    = juce::dsp::Convolution::Stereo;

    Convolution() { _formatManager.registerBasicFormats(); }

    auto prepare(juce::dsp::ProcessSpec const& spec) -> void
    {
        _spec = spec;

        if (_impulse.has_value()) {
            update();
        }
    }

    auto reset() -> void
    {
        _convolvers.clear();
        _filter = {};
    }

    template<typename ProcessContext>
        requires std::same_as<typename ProcessContext::SampleType, float>
    auto process(ProcessContext const& context) -> void
    {
        auto input  = context.getInputBlock();
        auto output = context.getOutputBlock();

        jassert(input.getNumSamples() == output.getNumSamples());
        jassert(input.getNumChannels() == output.getNumChannels());

        if constexpr (ProcessContext::usesSeparateInputAndOutputBlocks()) {
            output.copyFrom(input);
        }

        if (_convolvers.empty() or not _spec) {
            return;
        }

        if (input.getNumSamples() != _spec->maximumBlockSize) {
            return;
        }

        jassert(juce::isPowerOfTwo(input.getNumSamples()));
        for (auto ch{0U}; ch < output.getNumChannels(); ++ch) {
            auto io = stdex::mdspan{output.getChannelPointer(ch), stdex::extents{output.getNumSamples()}};
            std::invoke(_convolvers[ch], io);
        }
    }

    auto loadImpulseResponse(juce::File const& file, Stereo stereo, Trim trim, size_t size, Normalise normalise) -> void
    {
        jassertquiet(size == 0);
        jassertquiet(trim == Trim::no);
        jassertquiet(normalise == Normalise::no);
        jassertquiet(stereo == Stereo::yes);

        auto audioFile = load(_formatManager, file);
        jassertquiet(audioFile.buffer.getNumChannels() == 2);

        if (audioFile.sampleRate > 0.0) {
            _impulse = std::move(audioFile);
        }

        update();
    }

private:
    auto update() -> void
    {
        if (not _spec) {
            return;
        }

        if (not _impulse) {
            return;
        }

        auto resampled = resample(*_impulse, _spec->sampleRate);
        auto array     = to_mdarray(resampled.buffer);

        _filter = neo::convolution::uniform_partition(array.to_mdspan(), _spec->maximumBlockSize);
        _convolvers.resize(_spec->numChannels);

        for (auto ch{0U}; ch < _spec->numChannels; ++ch) {
            auto channel = stdex::submdspan(_filter.to_mdspan(), ch, stdex::full_extent, stdex::full_extent);
            _convolvers[ch].filter(channel);
        }
    }

    juce::AudioFormatManager _formatManager;

    std::optional<juce::dsp::ProcessSpec> _spec;
    std::optional<BufferWithSampleRate<float>> _impulse;

    stdex::mdarray<std::complex<float>, stdex::dextents<std::size_t, 3>> _filter;
    std::vector<neo::convolution::split_upols_convolver<std::complex<float>>> _convolvers;
};

}  // namespace neo
