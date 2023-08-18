#include "DenseConvolution.hpp"

#include "dsp/AudioBuffer.hpp"

#include <neo/fft/convolution/uniform_partition.hpp>
#include <neo/fft/transform/fftfreq.hpp>
#include <neo/math/a_weighting.hpp>
#include <neo/math/decibel.hpp>
#include <neo/math/next_power_of_two.hpp>

namespace neo {

static auto uniform_partition(juce::AudioBuffer<float> const& buffer, std::integral auto blockSize)
    -> stdex::mdarray<std::complex<float>, stdex::dextents<size_t, 3>>
{
    auto matrix = to_mdarray(buffer);
    return neo::fft::uniform_partition(matrix.to_mdspan(), static_cast<std::size_t>(blockSize));
}

DenseConvolution::DenseConvolution(int blockSize)
    : ConstantOverlapAdd<float>{neo::ilog2(juce::nextPowerOfTwo(blockSize)), 0}
{}

auto DenseConvolution::loadImpulseResponse(std::unique_ptr<juce::InputStream> stream) -> void
{
    jassert(stream != nullptr);

    auto formatManager = juce::AudioFormatManager{};
    formatManager.registerBasicFormats();

    auto reader = std::unique_ptr<juce::AudioFormatReader>(formatManager.createReaderFor(std::move(stream)));
    if (reader == nullptr) {
        return;
    }

    auto buffer = juce::AudioBuffer<float>{
        static_cast<int>(reader->numChannels),
        static_cast<int>(reader->lengthInSamples),
    };
    if (not reader->read(buffer.getArrayOfWritePointers(), buffer.getNumChannels(), 0, buffer.getNumSamples())) {
        return;
    }

    _impulse = ImpulseResponse{
        .buffer     = std::move(buffer),
        .sampleRate = reader->sampleRate,
    };
}

auto DenseConvolution::prepareFrame(juce::dsp::ProcessSpec const& spec) -> void
{
    jassert(juce::isPowerOfTwo(spec.maximumBlockSize));
    _spec = spec;
    updateImpulseResponse();
}

auto DenseConvolution::processFrame(juce::dsp::ProcessContextReplacing<float> const& context) -> void
{
    auto block = context.getOutputBlock();

    jassert(_spec.has_value());
    jassert(_spec->numChannels == block.getNumChannels());
    jassert(_convolvers.size() == block.getNumChannels());

    for (auto ch{0U}; ch < block.getNumChannels(); ++ch) {
        auto io = stdex::mdspan{block.getChannelPointer(ch), stdex::extents{block.getNumSamples()}};
        std::invoke(_convolvers[ch], io);
    }
}

auto DenseConvolution::resetFrame() -> void {}

auto DenseConvolution::updateImpulseResponse() -> void
{
    jassert(_spec.has_value());

    if (_impulse.has_value()) {
        _convolvers.resize(_spec->numChannels);
        auto resampled = resample(_impulse->buffer, _impulse->sampleRate, _spec->sampleRate);
        juce_normalization(resampled);
        _filter = uniform_partition(resampled, _spec->maximumBlockSize);
        for (auto ch{0U}; ch < _spec->numChannels; ++ch) {
            auto channel = stdex::submdspan(_filter.to_mdspan(), ch, stdex::full_extent, stdex::full_extent);
            _convolvers[ch].filter(channel);
        }
    } else {
        // jassertfalse;
        auto const identity = neo::generate_identity_impulse<float>(_spec->maximumBlockSize, 2);
        _filter             = stdex::mdarray<std::complex<float>, stdex::dextents<size_t, 3>>{
            _spec->numChannels,
            identity.extent(0),
            identity.extent(1),
        };
        _convolvers.resize(_spec->numChannels);
        for (auto ch{0U}; ch < _spec->numChannels; ++ch) {
            auto channel = stdex::submdspan(_filter.to_mdspan(), ch, stdex::full_extent, stdex::full_extent);
            neo::copy(identity.to_mdspan(), channel);
            _convolvers[ch].filter(channel);
        }
    }
}

auto dense_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>
{
    auto const blockSize = 512;

    auto output     = juce::AudioBuffer<float>{signal.getNumChannels(), signal.getNumSamples()};
    auto block      = std::vector<float>(size_t(blockSize));
    auto partitions = uniform_partition(filter, blockSize);

    for (auto ch{0}; ch < signal.getNumChannels(); ++ch) {
        auto convolver     = neo::fft::upols_convolver<float>{};
        auto const channel = static_cast<size_t>(ch);
        auto const full    = stdex::full_extent;
        convolver.filter(stdex::submdspan(partitions.to_mdspan(), channel, full, full));

        auto const* const in = signal.getReadPointer(ch);
        auto* const out      = output.getWritePointer(ch);

        for (auto i{0}; i < output.getNumSamples(); i += blockSize) {
            auto const numSamples = std::min(output.getNumSamples() - i, blockSize);
            std::fill(block.begin(), block.end(), 0.0F);
            std::copy(std::next(in, i), std::next(in, i + numSamples), block.begin());
            convolver(stdex::mdspan{block.data(), stdex::extents{block.size()}});
            std::copy(block.begin(), std::next(block.begin(), numSamples), std::next(out, i));
        }
    }

    return output;
}

[[nodiscard]] static auto
normalization_factor(stdex::mdspan<std::complex<float> const, stdex::dextents<size_t, 2>> filter) -> float
{
    auto maxPower = 0.0F;
    for (auto p{0UL}; p < filter.extent(0); ++p) {
        auto const partition = stdex::submdspan(filter, p, stdex::full_extent);
        for (auto bin{0UL}; bin < filter.extent(1); ++bin) {
            auto const amplitude = std::abs(partition(bin));
            maxPower             = std::max(maxPower, amplitude * amplitude);
        }
    }
    return 1.0F / maxPower;
}

auto sparse_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter, float thresholdDB)
    -> juce::AudioBuffer<float>
{
    auto const blockSize = 512;

    auto output     = juce::AudioBuffer<float>{signal.getNumChannels(), signal.getNumSamples()};
    auto block      = std::vector<float>(size_t(blockSize));
    auto partitions = uniform_partition(filter, blockSize);

    auto const K = neo::next_power_of_two((partitions.extent(2) - 1U) * 2U);

    auto const weights = [K, bins = partitions.extent(2)] {
        auto w = std::vector<float>(bins);
        for (auto i{0U}; i < w.size(); ++i) {
            auto const frequency = neo::fftfreq<float>(K, i, 44'100.0);
            auto const weight    = frequency > 0.0F ? neo::a_weighting(frequency) : 0.0F;

            w[i] = weight;
        }
        return w;
    }();

    for (auto ch{0}; ch < signal.getNumChannels(); ++ch) {
        auto convolver               = neo::fft::sparse_upols_convolver<float>{};
        auto const channel           = static_cast<size_t>(ch);
        auto const full              = stdex::full_extent;
        auto const channelPartitions = stdex::submdspan(partitions.to_mdspan(), channel, full, full);

        auto const scale = normalization_factor(channelPartitions);

        auto const isAboveThreshold = [thresholdDB, scale, &weights](auto /*row*/, auto col, auto bin) {
            auto const gain  = std::abs(bin);
            auto const power = gain * gain;
            auto const dB    = to_decibels(power * scale) + weights[col];
            return dB > thresholdDB;
        };

        convolver.filter(channelPartitions, isAboveThreshold);

        auto const* const in = signal.getReadPointer(ch);
        auto* const out      = output.getWritePointer(ch);

        for (auto i{0}; i < output.getNumSamples(); i += blockSize) {
            auto const numSamples = std::min(output.getNumSamples() - i, blockSize);
            std::fill(block.begin(), block.end(), 0.0F);
            std::copy(std::next(in, i), std::next(in, i + numSamples), block.begin());
            convolver(stdex::mdspan{block.data(), stdex::extents{block.size()}});
            std::copy(block.begin(), std::next(block.begin(), numSamples), std::next(out, i));
        }
    }

    return output;
}

}  // namespace neo