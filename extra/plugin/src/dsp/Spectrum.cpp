// SPDX-License-Identifier: MIT

#include "Spectrum.hpp"

#include "dsp/AudioBuffer.hpp"

#include <neo/unit/decibel.hpp>

#include <juce_dsp/juce_dsp.h>

namespace neo {

auto powerSpectrumImage(
    stdex::mdspan<std::complex<float> const, stdex::dextents<size_t, 2>> frames,
    std::function<float(std::size_t)> const& weighting,
    float threshold
) -> juce::Image
{
    auto const scale = [=] {
        auto max = 0.0F;
        for (auto frameIdx{0zu}; frameIdx < frames.extent(0); ++frameIdx) {
            for (auto binIdx{0zu}; binIdx < frames.extent(1); ++binIdx) {
                auto bin = std::abs(frames(frameIdx, binIdx));
                max      = std::max(max, bin * bin);
            }
        }
        return 1.0F / max;
    }();

    auto const fillPixel = [=](auto& img, int x, int y, auto bin) {
        auto const weight     = weighting(static_cast<std::size_t>(y));
        auto const power      = bin * bin;
        auto const normalized = power * scale;
        auto const dB         = neo::amplitude_to_db(normalized, -144.0F) * 0.5F + weight;
        auto const color      = dB < threshold ? juce::Colours::white : juce::Colours::black;
        img.setPixelAt(x, y, color);
    };

    auto const numFrames = static_cast<int>(frames.extent(0));
    auto const numBins   = static_cast<int>(frames.extent(1));

    auto img = juce::Image{juce::Image::PixelFormat::ARGB, numFrames, numBins, true};

    for (auto frameIdx{0}; frameIdx < numFrames; ++frameIdx) {
        for (auto binIdx{0}; binIdx < numBins; ++binIdx) {
            auto const bin = frames(frameIdx, binIdx);
            fillPixel(img, frameIdx, numBins - binIdx - 1, std::abs(bin));
        }
    }

    return img;
}

auto powerHistogram(
    stdex::mdspan<std::complex<float> const, stdex::dextents<size_t, 2>> spectogram,
    std::function<float(std::size_t)> const& weighting
) -> std::vector<int>
{

    auto const scale = [=] {
        auto max = 0.0F;
        for (auto frameIdx{0zu}; frameIdx < spectogram.extent(0); ++frameIdx) {
            for (auto binIdx{0zu}; binIdx < spectogram.extent(1); ++binIdx) {
                auto const bin = std::abs(spectogram(frameIdx, binIdx));
                max            = std::max(max, bin * bin);
            }
        }
        return 1.0F / max;
    }();

    auto histogram = std::vector<int>(144, 0);

    for (auto frameIdx{0zu}; frameIdx < spectogram.extent(0); ++frameIdx) {
        for (auto binIdx{0zu}; binIdx < spectogram.extent(1); ++binIdx) {
            auto const weight     = weighting(binIdx);
            auto const bin        = std::abs(spectogram(frameIdx, binIdx));
            auto const power      = bin * bin;
            auto const normalized = power * scale;
            auto const dB         = neo::amplitude_to_db(normalized, -144.0F) * 0.5F + weight;
            auto const dBClamped  = std::clamp(dB, -143.0F, 0.0F);
            auto const index      = static_cast<std::size_t>(juce::roundToInt(std::abs(dBClamped)));
            histogram[index] += 1;
        }
    }

    return histogram;
}

auto powerHistogramImage(
    stdex::mdspan<std::complex<float> const, stdex::dextents<size_t, 2>> spectogram,
    std::function<float(std::size_t)> const& weighting
) -> juce::Image
{
    auto const histogram = powerHistogram(spectogram, weighting);
    auto const maxBin    = std::max_element(histogram.begin(), std::prev(histogram.end()));
    if (maxBin == histogram.end()) {
        return {};
    }

    auto const binWidth  = 8;
    auto const imgWidth  = static_cast<int>((histogram.size() - 1zu) * binWidth);
    auto const imgHeight = 250;

    auto img = juce::Image{juce::Image::PixelFormat::ARGB, imgWidth, imgHeight, true};
    auto g   = juce::Graphics{img};
    g.fillAll(juce::Colours::black);

    for (auto i{0zu}; i < histogram.size() - 1zu; ++i) {
        auto const x      = int(i) * binWidth;
        auto const height = float(histogram[i]) / float(*maxBin) * float(imgHeight);
        auto const rect   = juce::Rectangle{x, 0, binWidth, juce::roundToInt(height)};

        g.setColour(juce::Colours::white);
        g.fillRect(rect.withBottomY(img.getBounds().getBottom()));

        if ((i == 32) || (i == 64) || (i == 96)) {
            g.setColour(juce::Colours::red.withAlpha(0.65F));
            g.fillRect(rect.withHeight(img.getHeight()));
        }
    }

    return img;
}

}  // namespace neo
