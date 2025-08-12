// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/complex/complex.hpp>
#include <neo/complex/split_complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/math/imag.hpp>
#include <neo/math/real.hpp>

#include <cassert>

namespace neo {

/// \ingroup neo-linalg
template<in_object InObj, out_object OutObj>
    requires(InObj::rank() == OutObj::rank())
constexpr auto copy(InObj in_obj, OutObj out_obj) noexcept -> void
{
    assert(detail::extents_equal(in_obj, out_obj));

    if constexpr (InObj::rank() == 1) {
        for (auto i{0ULL}; i < in_obj.extent(0); ++i) {
            out_obj[i] = in_obj[i];
        }
    } else {
        for (auto i{0ULL}; i < in_obj.extent(0); ++i) {
            for (auto j{0ULL}; j < in_obj.extent(1); ++j) {
                out_obj(i, j) = in_obj(i, j);
            }
        }
    }
}

template<in_vector InVec, out_vector OutVec>
    requires complex<value_type_t<InVec>>
constexpr auto copy(InVec in, split_complex<OutVec> out) noexcept -> void
{
    assert(detail::extents_equal(in, out.real, out.imag));

    for (auto i = std::size_t(0); i < static_cast<std::size_t>(in.extent(0)); ++i) {
        out.real[i] = static_cast<value_type_t<OutVec>>(math::real(in[i]));
        out.imag[i] = static_cast<value_type_t<OutVec>>(math::imag(in[i]));
    }
}

template<in_vector InVec, out_vector OutVec>
    requires complex<value_type_t<OutVec>>
constexpr auto copy(split_complex<InVec> in, OutVec out) noexcept -> void
{
    assert(detail::extents_equal(in.real, in.imag, out));

    using Complex = value_type_t<OutVec>;

    for (auto i = std::size_t(0); i < static_cast<std::size_t>(in.extent(0)); ++i) {
        out[i] = Complex{in.real[i], in.imag[i]};
    }
}

}  // namespace neo
