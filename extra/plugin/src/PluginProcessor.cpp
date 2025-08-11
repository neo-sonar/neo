// SPDX-License-Identifier: MIT

#include "PluginProcessor.hpp"

#include "PluginEditor.hpp"
#include "PluginParameter.hpp"

namespace neo {

PluginProcessor::PluginProcessor()
    : AudioProcessor(
          BusesProperties()
              .withInput("Input", juce::AudioChannelSet::stereo(), true)
              .withOutput("Output", juce::AudioChannelSet::stereo(), true)
      )
    , _valueTree{*this, nullptr, juce::Identifier("PerceptualConvolution"), neo::createParameters()}
    , _convolution{_valueTree}
{}

PluginProcessor::~PluginProcessor() = default;

auto PluginProcessor::getName() const -> juce::String const { return JucePlugin_Name; }

auto PluginProcessor::acceptsMidi() const -> bool { return false; }

auto PluginProcessor::producesMidi() const -> bool { return false; }

auto PluginProcessor::isMidiEffect() const -> bool { return false; }

auto PluginProcessor::getTailLengthSeconds() const -> double { return 0.0; }

auto PluginProcessor::getNumPrograms() -> int { return 1; }

auto PluginProcessor::getCurrentProgram() -> int { return 0; }

auto PluginProcessor::setCurrentProgram(int index) -> void { juce::ignoreUnused(index); }

auto PluginProcessor::getProgramName(int index) -> juce::String const
{
    juce::ignoreUnused(index);
    return {};
}

auto PluginProcessor::changeProgramName(int index, juce::String const& newName) -> void
{
    juce::ignoreUnused(index, newName);
}

auto PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) -> void
{
    _spec = juce::dsp::ProcessSpec{
        sampleRate,
        static_cast<std::uint32_t>(samplesPerBlock),
        static_cast<std::uint32_t>(getMainBusNumOutputChannels()),
    };

    _convolution.prepare(*_spec);

    _specListeners.call(&ProcessSpecListener::processSpecChanged, *_spec);
}

auto PluginProcessor::releaseResources() -> void { _convolution.reset(); }

auto PluginProcessor::isBusesLayoutSupported(BusesLayout const& layouts) const -> bool
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo()) {
        return false;
    }

    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet()) {
        return false;
    }

    return true;
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& /*midiMessages*/)
{
    juce::ScopedNoDenormals const noDenormals;

    for (auto i = getTotalNumInputChannels(); i < getTotalNumOutputChannels(); ++i) {
        buffer.clear(i, 0, buffer.getNumSamples());
    }

    auto block = juce::dsp::AudioBlock<float>{buffer};
    _convolution.process(block);
}

auto PluginProcessor::hasEditor() const -> bool { return true; }

auto PluginProcessor::createEditor() -> juce::AudioProcessorEditor* { return new PluginEditor(*this); }

auto PluginProcessor::getStateInformation(juce::MemoryBlock& destData) -> void
{
    juce::MemoryOutputStream stream(destData, false);
    _valueTree.state.writeToStream(stream);
}

auto PluginProcessor::setStateInformation(void const* data, int sizeInBytes) -> void
{
    juce::ValueTree const tree = juce::ValueTree::readFromData(data, static_cast<size_t>(sizeInBytes));
    jassert(tree.isValid());
    if (tree.isValid()) {
        _valueTree.state = tree;
    }
}

auto PluginProcessor::getProcessSpec() const noexcept -> juce::dsp::ProcessSpec
{
    return _spec.value_or(juce::dsp::ProcessSpec{44'100.0, 512, 2});
}

auto PluginProcessor::addProcessSpecListener(ProcessSpecListener* listener) -> void { _specListeners.add(listener); }

auto PluginProcessor::removeProcessSpecListener(ProcessSpecListener* listener) -> void
{
    _specListeners.remove(listener);
}

auto PluginProcessor::getState() noexcept -> juce::AudioProcessorValueTreeState& { return _valueTree; }

auto PluginProcessor::getState() const noexcept -> juce::AudioProcessorValueTreeState const& { return _valueTree; }

}  // namespace neo

// This creates new instances of the plugin..
auto JUCE_CALLTYPE createPluginFilter() -> juce::AudioProcessor* { return new neo::PluginProcessor{}; }
