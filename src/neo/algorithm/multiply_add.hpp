#pragma once

#include <neo/config.hpp>

#include <neo/complex/split_complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>

#if defined(NEO_PLATFORM_APPLE)
    #include <Accelerate/Accelerate.h>
#endif

#include <cassert>
#include <utility>

namespace neo {

// out = x * y + z
template<in_vector VecX, in_vector VecY, in_vector VecZ, out_vector VecOut>
constexpr auto multiply_add(VecX x, VecY y, VecZ z, VecOut out) noexcept -> void
{
    assert(detail::extents_equal(x, y, z, out));

    for (decltype(x.extent(0)) i{0}; i < x.extent(0); ++i) {
        out[i] = x[i] * y[i] + z[i];
    }
}

// out = x * y + z
template<typename U, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_add(
    in_vector auto x,
    sparse_matrix<U, IndexType, ValueContainer, IndexContainer> const& y,
    typename sparse_matrix<U, IndexType, ValueContainer, IndexContainer>::index_type y_row,
    in_vector auto z,
    out_vector auto out
) noexcept -> void
{
    assert(x.extent(0) == y.columns());

    auto const& rrows = y.row_container();
    auto const& rcols = y.column_container();
    auto const& rvals = y.value_container();

    for (auto i{rrows[y_row]}; i < rrows[y_row + 1]; ++i) {
        auto col = rcols[i];
        out[col] = x[col] * rvals[i] + z[col];
    }
}

// out = x * y + z
template<in_vector VecX, in_vector VecY, in_vector VecZ, out_vector VecOut>
constexpr auto
multiply_add(split_complex<VecX> x, split_complex<VecY> y, split_complex<VecZ> z, split_complex<VecOut> out) noexcept
    -> void
{
    assert(detail::extents_equal(x.real, x.imag, y.real, y.imag, z.real, z.imag, out.real, out.imag));

#if defined(NEO_PLATFORM_APPLE)
    if constexpr (detail::all_same_value_type_v<VecX, VecY, VecZ, VecOut>) {
        if (detail::strides_equal_to<1>(x.real, x.imag, y.real, y.imag, z.real, z.imag, out.real, out.imag)) {
            using Float = typename VecX::value_type;
            if constexpr (std::same_as<Float, float> or std::same_as<Float, double>) {
                using split_t = std::conditional_t<std::same_as<Float, float>, DSPSplitComplex, DSPDoubleSplitComplex>;
                auto xsc = split_t{.realp = const_cast<Float*>(&x.real[0]), .imagp = const_cast<Float*>(&x.imag[0])};
                auto ysc = split_t{.realp = const_cast<Float*>(&y.real[0]), .imagp = const_cast<Float*>(&y.imag[0])};
                auto zsc = split_t{.realp = const_cast<Float*>(&z.real[0]), .imagp = const_cast<Float*>(&z.imag[0])};
                auto osc = split_t{.realp = &out.real[0], .imagp = &out.imag[0]};

                if constexpr (std::same_as<Float, float>) {
                    vDSP_zvma(&xsc, 1, &ysc, 1, &zsc, 1, &osc, 1, x.real.extent(0));
                } else {
                    vDSP_zvmaD(&xsc, 1, &ysc, 1, &zsc, 1, &osc, 1, x.real.extent(0));
                }

                return;
            }
        }
    }
#endif

    for (auto i{0}; i < static_cast<int>(x.real.extent(0)); ++i) {
        auto const xre = x.real[i];
        auto const xim = x.imag[i];
        auto const yre = y.real[i];
        auto const yim = y.imag[i];

        out.real[i] = (xre * yre - xim * yim) + z.real[i];
        out.imag[i] = (xre * yim + xim * yre) + z.imag[i];
    }
}

}  // namespace neo