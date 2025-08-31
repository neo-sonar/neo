// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/compressed_accessor.hpp>
#include <neo/container/mdspan.hpp>

#include <cmath>
#include <limits>

namespace neo::convolution {

/// \ingroup neo-convolution
template<complex FloatComplex, complex IntComplex>
struct compressed_fdl
{
    using value_type      = FloatComplex;
    using compressed_type = IntComplex;

    compressed_fdl() = default;

    explicit compressed_fdl(stdex::dextents<size_t, 2> extents) : _fdl{extents} {}

    [[nodiscard]] auto operator[](std::integral auto index) const noexcept -> in_vector_of<FloatComplex> auto
    {
        auto const subfilter = stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent);
        return stdex::mdspan{
            subfilter.data_handle(),
            subfilter.mapping(),
            compressed_accessor<FloatComplex, typename decltype(subfilter)::accessor_type>{subfilter.accessor()},
        };
    }

    auto insert(in_vector_of<FloatComplex> auto input, std::integral auto index) noexcept -> void
    {
        auto const compress = [](auto val) {
            using int_type         = typename IntComplex::value_type;
            constexpr auto max_val = std::numeric_limits<int_type>::max();
            return static_cast<int_type>(std::lround(val * static_cast<float>(max_val)));
        };

        auto const fdl = stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent);
        for (auto i{0zu}; i < input.extent(0); ++i) {
            fdl[i] = IntComplex{compress(input[i].real()), compress(input[i].imag())};
        }
    }

private:
    stdex::mdarray<IntComplex, stdex::dextents<size_t, 2>> _fdl{};
};

}  // namespace neo::convolution
