// SPDX-License-Identifier: MIT

#pragma once

#include "PluginProcessor.hpp"
#include "dsp/AudioBuffer.hpp"
#include "dsp/AudioFile.hpp"

#include <neo/container/mdspan.hpp>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include <chrono>
#include <complex>

namespace neo {

struct BenchmarkTab final
    : juce::Component
    , juce::Value::Listener
    , ProcessSpecListener

{
    explicit BenchmarkTab(PluginProcessor& processor, juce::AudioFormatManager& formatManager);
    ~BenchmarkTab() override;

    auto setImpulseResponseFile(juce::File const& file) -> void;

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;
    auto valueChanged(juce::Value& value) -> void override;
    auto processSpecChanged(juce::dsp::ProcessSpec const& spec) -> void override;

private:
    auto selectSignalFile() -> void;
    auto loadSignalFile(juce::File const& file) -> void;

    auto runBenchmarks() -> void;
    auto runWeightingTests() -> void;
    auto runJuceConvolutionBenchmark() -> void;
    auto runNeoConvolutionBenchmark() -> void;
    auto runDenseConvolutionBenchmark() -> void;
    auto runSparseConvolverBenchmark() -> void;
    auto runConvolverTests() -> void;
    auto runSparseQualityTests() -> void;

    template<typename Convolver>
    auto runDenseBenchmark(juce::String const& name) -> void
    {
        auto start     = std::chrono::system_clock::now();
        auto result    = neo::dense_convolve<Convolver>(_signal.buffer, _impulse.buffer, 4096);
        auto const end = std::chrono::system_clock::now();

        auto output = to_mdarray(result);
        neo::normalize_peak(output.to_mdspan());

        auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        juce::MessageManager::callAsync([this, elapsed, name] {
            _fileInfo.moveCaretToEnd(false);
            _fileInfo.insertTextAtCaret(name + ": " + juce::String{elapsed.count()} + "\n");
        });

        auto file = getBenchmarkResultsDirectory().getNonexistentChildFile(name, ".wav");
        writeToWavFile(file, output, _spec.sampleRate, 32);
    }

    auto updateImages() -> void;

    [[nodiscard]] static auto getBenchmarkResultsDirectory() -> juce::File;

    PluginProcessor& _processor;
    juce::AudioFormatManager& _formatManager;
    juce::dsp::ProcessSpec _spec{};

    juce::TextButton _selectSignalFile{"Select Signal File"};
    juce::TextButton _render{"Render"};
    juce::PropertyPanel _propertyPanel{};
    juce::TextEditor _fileInfo{};
    juce::ImageComponent _spectogramImage{};
    juce::ImageComponent _histogramImage{};

    juce::File _signalFile{};
    juce::File _impulseFile{};
    BufferWithSampleRate<float> _signal{};
    BufferWithSampleRate<float> _impulse{};

    stdex::mdarray<std::complex<float>, stdex::dextents<size_t, 3>> _spectrum;
    stdex::mdarray<std::complex<float>, stdex::dextents<size_t, 3>> _partitions;

    // Sparsity
    juce::Value _binsToKeep{juce::var{10}};
    juce::Value _dynamicRange{juce::var{90.0F}};
    juce::Value _weighting{juce::var{"A-Weighting"}};
    juce::Value _engine{juce::Array<juce::var>{juce::var{"dense"}}};

    // Quality
    juce::Value _stftWindowSize{juce::var{512}};

    std::unique_ptr<juce::FileChooser> _fileChooser{nullptr};
    juce::ThreadPool _threadPool{juce::ThreadPoolOptions{}.withNumberOfThreads(1).withThreadName("Benchmark Thread")};
};

}  // namespace neo
