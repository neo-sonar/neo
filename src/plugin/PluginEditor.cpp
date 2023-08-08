#include "PluginEditor.hpp"

#include "dsp/convolution.hpp"
#include "dsp/normalize.hpp"
#include "dsp/render.hpp"
#include "dsp/resample.hpp"
#include "dsp/spectogram.hpp"
#include "dsp/stft.hpp"
#include "dsp/wav.hpp"
#include "neo/convolution/container/sparse_matrix.hpp"

#include <span>

namespace neo {

namespace {

struct JuceConvolver
{
    explicit JuceConvolver(juce::File impulse, juce::dsp::ProcessSpec const& spec)
        : _impulse{std::move(impulse)}
        , _spec{spec}
    {
        auto const trim      = juce::dsp::Convolution::Trim::no;
        auto const stereo    = juce::dsp::Convolution::Stereo::yes;
        auto const normalize = juce::dsp::Convolution::Normalise::no;

        _convolver.prepare(spec);
        _convolver.loadImpulseResponse(_impulse, stereo, trim, 0, normalize);

        // impulse is loaded on background thread, may not have loaded fast enough in
        // unit-tests
        std::this_thread::sleep_for(std::chrono::milliseconds{2000});
    }

    auto prepare(juce::dsp::ProcessSpec const& spec) -> void { jassertquiet(_spec == spec); }

    auto reset() -> void { _convolver.reset(); }

    template<typename Context>
    auto process(Context const& context) -> void
    {
        _convolver.process(context);
    }

private:
    juce::File _impulse;
    juce::dsp::ConvolutionMessageQueue _queue;
    juce::dsp::Convolution _convolver{juce::dsp::Convolution::Latency{0}, _queue};
    juce::dsp::ProcessSpec _spec;
};

template<typename T, typename U, typename V>
[[nodiscard]] constexpr auto frequencyForBin(U windowSize, V index, double sampleRate) -> T
{
    static_assert(std::is_integral_v<U>);
    static_assert(std::is_integral_v<V>);

    return static_cast<T>(index) * static_cast<T>(sampleRate) / static_cast<T>(windowSize);
}

}  // namespace

PluginEditor::PluginEditor(PluginProcessor& p) : AudioProcessorEditor(&p)
{
    _formats.registerBasicFormats();

    _openFile.onClick      = [this] { openFile(); };
    _runBenchmarks.onClick = [this] { runBenchmarks(); };

    _threshold.setRange({-144.0, -10.0}, 0.0);
    _threshold.setValue(-90.0, juce::dontSendNotification);
    _threshold.onDragEnd = [this] { updateImages(); };

    _peakOrPower.setClickingTogglesState(true);
    _peakOrPower.setToggleState(true, juce::dontSendNotification);
    _peakOrPower.onClick = [this] {
        _peakOrPower.setButtonText(_peakOrPower.getToggleState() ? "Power" : "Peak");
        updateImages();
    };

    _weighting.setClickingTogglesState(true);
    _weighting.setToggleState(true, juce::dontSendNotification);
    _weighting.onClick = [this] { updateImages(); };

    _fileInfo.setReadOnly(true);
    _fileInfo.setMultiLine(true);
    _spectogramImage.setImagePlacement(juce::RectanglePlacement::centred);
    _histogramImage.setImagePlacement(juce::RectanglePlacement::centred);
    _tooltipWindow->setMillisecondsBeforeTipAppears(750);

    addAndMakeVisible(_openFile);
    addAndMakeVisible(_runBenchmarks);
    addAndMakeVisible(_threshold);
    addAndMakeVisible(_peakOrPower);
    addAndMakeVisible(_weighting);
    addAndMakeVisible(_fileInfo);
    addAndMakeVisible(_spectogramImage);
    addAndMakeVisible(_histogramImage);

    setResizable(true, true);
    setSize(600, 400);
}

PluginEditor::~PluginEditor() noexcept { setLookAndFeel(nullptr); }

auto PluginEditor::paint(juce::Graphics& g) -> void
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

auto PluginEditor::resized() -> void
{
    auto bounds = getLocalBounds();

    auto controls = bounds.removeFromTop(bounds.proportionOfHeight(0.1));
    auto width    = controls.proportionOfWidth(0.2);
    _openFile.setBounds(controls.removeFromLeft(width));
    _runBenchmarks.setBounds(controls.removeFromLeft(width));
    _peakOrPower.setBounds(controls.removeFromLeft(width));
    _weighting.setBounds(controls.removeFromLeft(width));
    _threshold.setBounds(controls);

    _fileInfo.setBounds(bounds.removeFromLeft(bounds.proportionOfWidth(0.2)));
    _spectogramImage.setBounds(bounds.removeFromLeft(bounds.proportionOfWidth(0.55)));
    _histogramImage.setBounds(bounds.reduced(4));
}

auto PluginEditor::openFile() -> void
{
    auto const* msg         = "Please select a impulse response";
    auto const homeDir      = juce::File::getSpecialLocation(juce::File::userMusicDirectory);
    auto const chooserFlags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    auto const load         = [this](juce::FileChooser const& chooser) {
        if (chooser.getResults().isEmpty()) { return; }

        auto const file     = chooser.getResult();
        auto const filename = file.getFileNameWithoutExtension();

        _impulse  = loadAndResample(_formats, file, 44'100.0);
        _spectrum = fft::stft(_impulse, 1024);

        _fileInfo.setText(filename + " (" + juce::String(_impulse.getNumSamples()) + ")");
        updateImages();

        repaint();
    };

    _fileChooser = std::make_unique<juce::FileChooser>(msg, homeDir, _formats.getWildcardForAllFormats());
    _fileChooser->launchAsync(chooserFlags, load);
}

auto PluginEditor::runBenchmarks() -> void
{
    _signalFile = juce::File{R"(C:\Users\tobias\Music\Loops\Drums.wav)"};
    _filterFile = juce::File{R"(C:\Users\tobias\Music\Samples\IR\LexiconPCM90 Halls\ORCH_gothic hall.WAV)"};

    _signal = loadAndResample(_formats, _signalFile, 44'100.0);
    _filter = loadAndResample(_formats, _filterFile, 44'100.0);

    // runDynamicRangeTests();
    // runJuceConvolverBenchmark();
    // runDenseConvolverBenchmark();
    // runDenseStereoConvolverBenchmark();
    // runSparseConvolverBenchmark();
}

auto PluginEditor::runWeightingTests() -> void
{
    auto normalized = _impulse;
    juce_normalization(normalized);

    auto const isPower     = _peakOrPower.getToggleState();
    auto const powerOrPeak = [isPower](auto value) { return isPower ? value * value : value; };

    auto const blockSize  = 512ULL;
    auto const impulse    = to_mdarray(normalized);
    auto const partitions = neo::fft::uniform_partition(impulse, blockSize);
    auto const scale      = [filter = partitions.to_mdspan(), powerOrPeak] {
        auto max = 0.0F;
        for (auto ch{0U}; ch < filter.extent(0); ++ch) {
            for (auto f{0U}; f < filter.extent(1); ++f) {
                for (auto b{0U}; b < filter.extent(2); ++b) {
                    max = std::max(max, powerOrPeak(std::abs(filter(ch, f, b))));
                }
            }
        }
        return 1.0F / max;
    }();

    auto const weighting = [=, this](std::size_t binIndex) {
        if (_weighting.getToggleState()) {
            auto const frequency = frequencyForBin<float>(blockSize * 2ULL, binIndex, 44'100.0);
            if (juce::exactlyEqual(frequency, 0.0F)) { return 0.0F; }
            auto const weight = neo::fft::a_weighting(frequency);
            return weight;
        }
        return 0.0F;
    };

    auto count           = 0;
    auto const threshold = static_cast<float>(_threshold.getValue());

    for (auto f{0U}; f < partitions.extent(1); ++f) {
        for (auto b{0U}; b < partitions.extent(2); ++b) {
            auto const weight = weighting(b);
            for (auto ch{0U}; ch < partitions.extent(0); ++ch) {
                auto const bin = powerOrPeak(std::abs(partitions(ch, f, b)));
                auto const dB  = juce::Decibels::gainToDecibels(bin * scale, -144.0F) + weight;
                count += static_cast<int>(dB > threshold);
            }
        }
    }

    auto const total = static_cast<double>(partitions.size());
    auto const line  = "\n" + juce::String(static_cast<double>(count) / total * 100.0, 2) + "% "
                    + juce::String(threshold, 1) + "dB " + (isPower ? juce::String("Power") : juce::String("Peak"))
                    + (_weighting.getToggleState() ? juce::String(" A-Weighting") : juce::String(" No weighting"));

    _fileInfo.moveCaretToEnd(false);
    _fileInfo.insertTextAtCaret(line);
}

auto PluginEditor::runDynamicRangeTests() -> void
{
    auto N          = 1024;
    auto normalized = _signal;
    juce_normalization(normalized);

    auto coeffs = neo::fft::stft(normalized, N);

    for (auto frame{0U}; frame < coeffs.extent(0); ++frame) {
        auto mins = std::array<float, 2>{999.0F, 999.0F};
        auto maxs = std::array<float, 2>{0.0F, 0.0F};

        for (auto bin{0U}; bin < coeffs.extent(1); ++bin) {
            auto const coeff = coeffs(frame, bin);

            mins[0] = std::min(mins[0], coeff.real());
            maxs[0] = std::max(maxs[0], coeff.real());

            mins[1] = std::min(mins[1], coeff.imag());
            maxs[1] = std::max(maxs[1], coeff.imag());
        }

        auto const realRange = std::abs(maxs[0] - mins[0]);
        auto const imagRange = std::abs(maxs[1] - mins[1]);
        auto const range     = std::max(realRange, imagRange);
        auto const scale     = 1.0F / range;

        std::printf("frame: %-3u range: %.6f scale: %.6f\n", frame, range, scale);
    }
}

auto PluginEditor::runJuceConvolverBenchmark() -> void
{
    auto proc = JuceConvolver{
        _filterFile,
        {44'100.0, 512, 2}
    };

    auto start  = std::chrono::system_clock::now();
    auto output = juce::AudioBuffer<float>{_signal.getNumChannels(), _signal.getNumSamples()};
    auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("jconv", ".wav");

    processBlocks(proc, _signal, output, 512, 44'100.0);
    peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    auto end = std::chrono::system_clock::now();
    std::cout << "JCONV: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';

    writeToWavFile(file, output, 44'100.0, 32);
}

auto PluginEditor::runDenseConvolverBenchmark() -> void
{
    auto start = std::chrono::system_clock::now();

    auto output = fft::dense_convolve(_signal, _filter);
    auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_dense", ".wav");

    peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    auto end = std::chrono::system_clock::now();
    std::cout << "TCONV-DENSE: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';

    writeToWavFile(file, output, 44'100.0, 32);
}

auto PluginEditor::runDenseStereoConvolverBenchmark() -> void
{
    auto start = std::chrono::system_clock::now();

    auto output = fft::dense_stereo_convolve(_signal, _filter);
    auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_dense_stereo", ".wav");

    peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    auto end = std::chrono::system_clock::now();
    std::cout << "TCONV-DENSE-STEREO: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << '\n';

    writeToWavFile(file, output, 44'100.0, 32);
}

auto PluginEditor::runSparseConvolverBenchmark() -> void
{
    auto start = std::chrono::system_clock::now();

    auto thresholdDB = -40.0F;
    auto output      = fft::sparse_convolve(_signal, _filter, thresholdDB);
    auto file        = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_sparse_40", ".wav");

    peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    auto end = std::chrono::system_clock::now();
    std::cout << "TCONV-SPARSE(40): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << '\n';

    writeToWavFile(file, output, 44'100.0, 32);
}

auto PluginEditor::updateImages() -> void
{
    if (_spectrum.size() == 0) { return; }

    auto const weighting = [this](std::size_t binIndex) {
        if (_weighting.getToggleState()) {
            auto const frequency = frequencyForBin<float>(1024, binIndex, 44'100.0);
            if (juce::exactlyEqual(frequency, 0.0F)) { return 0.0F; }
            auto const weight = neo::fft::a_weighting(frequency);
            return weight;
        }
        return 0.0F;
    };

    auto spectogramImage = fft::powerSpectrumImage(_spectrum, weighting, static_cast<float>(_threshold.getValue()));
    auto histogramImage  = fft::powerHistogramImage(_spectrum, weighting);

    _spectogramImage.setImage(spectogramImage);
    _histogramImage.setImage(histogramImage);

    runWeightingTests();
    repaint();
}

}  // namespace neo
