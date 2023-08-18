#include "Spectrum.hpp"

#include "dsp/AudioBuffer.hpp"

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
        for (auto f{0U}; f < frames.extent(0); ++f) {
            for (auto b{0U}; b < frames.extent(1); ++b) {
                auto const bin = std::abs(frames(f, b));
                max            = std::max(max, bin * bin);
            }
        }
        return 1.0F / max;
    }();

    auto const fillPixel = [=](auto& img, int x, int y, auto bin) {
        auto const weight     = weighting(static_cast<std::size_t>(y));
        auto const power      = bin * bin;
        auto const normalized = power * scale;
        auto const dB         = juce::Decibels::gainToDecibels(normalized, -144.0F) + weight;
        auto const dBClamped  = std::clamp(dB, -144.0F, 0.0F);
        auto const color      = [=] {
            if (dB < threshold) {
                auto level = juce::jmap(dBClamped, -144.0F, threshold, 0.0F, 1.0F);
                return juce::Colours::white.darker(level);
            }
            return juce::Colours::black;
        }();

        img.setPixelAt(x, y, color);
    };

    auto const rows = static_cast<int>(frames.extent(0));
    auto const cols = static_cast<int>(frames.extent(1));
    auto img        = juce::Image{juce::Image::PixelFormat::ARGB, cols, rows, true};

    for (auto f{0U}; f < frames.extent(0); ++f) {
        for (auto b{0U}; b < frames.extent(1); ++b) {
            fillPixel(img, static_cast<int>(b), static_cast<int>(f), std::abs(frames(f, b)));
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
        for (auto f{0U}; f < spectogram.extent(0); ++f) {
            for (auto b{0U}; b < spectogram.extent(1); ++b) {
                auto const bin = std::abs(spectogram(f, b));
                max            = std::max(max, bin * bin);
            }
        }
        return 1.0F / max;
    }();

    auto histogram = std::vector<int>(144, 0);

    for (auto f{0U}; f < spectogram.extent(0); ++f) {
        for (auto b{0U}; b < spectogram.extent(1); ++b) {
            auto const weight     = weighting(b);
            auto const bin        = std::abs(spectogram(f, b));
            auto const power      = bin * bin;
            auto const normalized = power * scale;
            auto const dB         = juce::Decibels::gainToDecibels(normalized, -144.0F) + weight;
            auto const dBClamped  = std::clamp(dB, -143.0F, 0.0F);
            auto const index      = static_cast<std::size_t>(std::lround(std::abs(dBClamped)));
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
    auto const imgWidth  = static_cast<int>((histogram.size() - 1) * binWidth);
    auto const imgHeight = 250;

    auto img = juce::Image{juce::Image::PixelFormat::ARGB, imgWidth, imgHeight, true};
    auto g   = juce::Graphics{img};
    g.fillAll(juce::Colours::black);

    for (auto i{0U}; i < histogram.size() - 1U; ++i) {
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