// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>

namespace neo::fft {

/// Reorder input using base-radix digit reversal permutation.
/// \ingroup neo-fft
template<std::size_t Radix>
struct digitrevorder_plan
{
    explicit digitrevorder_plan(std::size_t size) : _lut{make(size)} {}

    template<inout_vector Vec>
        requires(complex<value_type_t<Vec>>)
    auto operator()(Vec x) noexcept -> void
    {
        for (auto i{0U}; i < _lut.size(); ++i) {
            if (i < _lut[i]) {
                std::ranges::swap(x[i], x[_lut[i]]);
            }
        }
    }

private:
    [[nodiscard]] static auto make(std::size_t size) -> std::vector<std::uint32_t>
    {
        auto lut = std::vector<std::uint32_t>(size);

        auto j = 0UL;
        for (auto i = 0UL; i < size - 1UL; i++) {
            lut[i] = static_cast<std::uint32_t>(j);

            auto k = (Radix - 1UL) * size / Radix;
            while (k <= j) {
                j -= k;
                k /= Radix;
            }
            j += k / (Radix - 1);
        }

        return lut;
    }

    std::vector<std::uint32_t> _lut;
};

}  // namespace neo::fft
