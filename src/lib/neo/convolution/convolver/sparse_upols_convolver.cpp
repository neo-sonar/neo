#include "sparse_upols_convolver.hpp"

#include "neo/convolution/math/next_power_of_two.hpp"

#include <functional>
#include <random>

namespace neo::fft {

static auto toDecibels(float gain) -> float
{
    if (gain == 0.0F) { return -144.0F; }
    return 20.0F * std::log10(gain);
}

static auto multiply_and_accumulate(
    Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<std::size_t, 2>> lhs,
    sparse_matrix<std::complex<float>> const& rhs,
    std::span<std::complex<float>> accumulator,
    std::size_t shift
)
{
    assert(lhs.extent(0) == rhs.rows());
    assert(lhs.extent(1) == rhs.columns());
    assert(lhs.extent(1) > 0);
    assert(shift < lhs.extent(0));

    multiply_elementwise_accumulate_columnwise(lhs, rhs, accumulator, shift);
}

static auto normalization_factor(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter) -> float
{
    auto maxPower = 0.0F;
    for (auto partition{0UL}; partition < filter.extent(0); ++partition) {
        for (auto bin{0UL}; bin < filter.extent(1); ++bin) {
            auto const gain = std::abs(filter(partition, bin));
            maxPower        = std::max(maxPower, gain * gain);
        }
    }
    return 1.0F / maxPower;
}

auto sparse_upols_convolver::filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter)
    -> void
{
    auto const K     = next_power_of_two((filter.extent(1) - 1U) * 2U);
    auto const scale = normalization_factor(filter);

    auto const isAboveThreshold = [this, scale, K](auto /*row*/, auto col, auto bin) {
        auto const frequency = neo::fft::frequency_for_bin<float>(K, col, 44'100.0);
        auto const weight    = frequency > 0.0F ? neo::fft::a_weighting(frequency) : 0.0F;

        auto const gain  = std::abs(bin);
        auto const power = gain * gain;
        auto const dB    = toDecibels(power * scale) + weight;
        return dB > _thresholdDB;
    };

    _fdl    = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _filter = sparse_matrix<std::complex<float>>{filter, isAboveThreshold};

    _rfft = std::make_unique<rfft_radix2_plan<float>>(ilog2(K));
    _window.resize(K);
    _rfftBuf.resize(_window.size());
    _irfftBuf.resize(_window.size());
    _accumulator.resize(_filter.columns());

    _fdlIndex = 0;
}

auto sparse_upols_convolver::operator()(std::span<float> block) -> void
{
    assert(block.size() * 2U == _window.size());

    auto const blockSize = std::ssize(block);

    // Time domain input buffer
    std::shift_left(_window.begin(), _window.end(), blockSize);
    std::copy(block.begin(), block.end(), std::prev(_window.end(), blockSize));

    // 2B-point R2C-FFT
    std::invoke(*_rfft, _window, _rfftBuf);

    // Copy to FDL
    for (auto i{0U}; i < _fdl.extent(1); ++i) { _fdl(_fdlIndex, i) = _rfftBuf[i] / float(_rfft->size()); }

    // DFT-spectrum additions
    std::fill(_accumulator.begin(), _accumulator.end(), 0.0F);
    multiply_and_accumulate(_fdl, _filter, _accumulator, _fdlIndex);

    // All contents (DFT spectra) in the FDL are shifted up by one slot.
    ++_fdlIndex;
    if (_fdlIndex == _fdl.extent(0)) { _fdlIndex = 0; }

    // 2B-point C2R-IFFT
    std::invoke(*_rfft, _accumulator, _irfftBuf);

    // Copy blockSize samples to output
    std::copy(std::prev(_irfftBuf.end(), blockSize), _irfftBuf.end(), block.begin());
}

}  // namespace neo::fft
