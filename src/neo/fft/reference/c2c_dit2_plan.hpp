// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/order.hpp>
#include <neo/fft/reference/bitrevorder.hpp>
#include <neo/fft/reference/kernel/c2c_dit2.hpp>
#include <neo/fft/twiddle.hpp>
#include <neo/math/polar.hpp>

#include <cassert>
#include <numbers>

namespace neo::fft {

/// \brief C2C Cooley–Tukey Radix-2 DIT
/// \ingroup neo-fft
template<typename Complex, typename Kernel = kernel::c2c_dit2_v3>
struct c2c_dit2_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    c2c_dit2_plan(from_order_tag /*tag*/, size_type order);

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type;
    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type;

    [[nodiscard]] auto order() const noexcept -> size_type;
    [[nodiscard]] auto size() const noexcept -> size_type;

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Complex>
    auto operator()(Vec x, direction dir) noexcept -> void;

private:
    [[nodiscard]] static auto check_order(size_type order) -> size_type;

    size_type _order;
    size_type _size{neo::ipow<2zu>(order())};
    bitrevorder_plan _reorder{static_cast<size_t>(_order)};
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _wf{
        make_twiddle_lut_radix2<Complex>(_size, direction::forward),
    };
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _wb{
        make_twiddle_lut_radix2<Complex>(_size, direction::backward),
    };
};

template<typename Complex, typename Kernel>
c2c_dit2_plan<Complex, Kernel>::c2c_dit2_plan(from_order_tag /*tag*/, size_type order) : _order{check_order(order)}
{}

template<typename Complex, typename Kernel>
constexpr auto c2c_dit2_plan<Complex, Kernel>::max_order() noexcept -> size_type
{
    return size_type{27};
}

template<typename Complex, typename Kernel>
constexpr auto c2c_dit2_plan<Complex, Kernel>::max_size() noexcept -> size_type
{
    return neo::ipow<2zu>(max_order());
}

template<typename Complex, typename Kernel>
auto c2c_dit2_plan<Complex, Kernel>::order() const noexcept -> size_type
{
    return _order;
}

template<typename Complex, typename Kernel>
auto c2c_dit2_plan<Complex, Kernel>::size() const noexcept -> size_type
{
    return _size;
}

template<typename Complex, typename Kernel>
template<inout_vector Vec>
    requires std::same_as<typename Vec::value_type, Complex>
auto c2c_dit2_plan<Complex, Kernel>::operator()(Vec x, direction dir) noexcept -> void
{
    assert(std::cmp_equal(x.size(), _size));

    _reorder(x);

    if (auto const kernel = Kernel{}; dir == direction::forward) {
        kernel(x, _wf.to_mdspan());
    } else {
        kernel(x, _wb.to_mdspan());
    }
}

template<typename Complex, typename Kernel>
auto c2c_dit2_plan<Complex, Kernel>::check_order(size_type order) -> size_type
{
    if (order > max_order()) {
        throw std::runtime_error{"fallback: unsupported order '" + std::to_string(int(order)) + "'"};
    }
    return order;
}

}  // namespace neo::fft
