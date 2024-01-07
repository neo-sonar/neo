// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/fallback/bitrevorder.hpp>
#include <neo/fft/fallback/conjugate_view.hpp>
#include <neo/fft/fallback/kernel/c2c_dit2.hpp>
#include <neo/fft/order.hpp>
#include <neo/math/polar.hpp>

#include <cassert>
#include <numbers>

namespace neo::fft {

namespace detail {

template<inout_vector OutVec>
auto fill_radix2_twiddles(OutVec table, direction dir = direction::forward) noexcept -> void
{
    using Complex = typename OutVec::value_type;
    using Float   = typename Complex::value_type;

    auto const table_size = table.size();
    auto const fft_size   = table_size * 2ULL;
    auto const sign       = dir == direction::forward ? Float(-1) : Float(1);
    auto const two_pi     = static_cast<Float>(std::numbers::pi * 2.0);

    for (std::size_t i = 0; i < table_size; ++i) {
        auto const angle   = sign * two_pi * Float(i) / Float(fft_size);
        auto const twiddle = math::polar(Float(1), angle);             // returns std::complex
        table[i]           = Complex{twiddle.real(), twiddle.imag()};  // convert to custom complex (maybe)
    }
}

template<complex Complex>
auto make_radix2_twiddles(std::size_t size, direction dir = direction::forward)
{
    auto table = stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>>{size / 2U};
    fill_radix2_twiddles(table.to_mdspan(), dir);
    return table;
}

}  // namespace detail

template<typename Complex, typename Kernel = kernel::c2c_dit2_v3>
struct fallback_fft_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit fallback_fft_plan(fft::order order);

    [[nodiscard]] static constexpr auto max_order() noexcept -> fft::order;
    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type;

    [[nodiscard]] auto order() const noexcept -> fft::order;
    [[nodiscard]] auto size() const noexcept -> size_type;

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Complex>
    auto operator()(Vec x, direction dir) noexcept -> void;

private:
    [[nodiscard]] static auto check_order(fft::order order) -> fft::order;

    fft::order _order;
    size_type _size{fft::size(order())};
    bitrevorder_plan _reorder{static_cast<size_t>(_order)};
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _twiddles{
        detail::make_radix2_twiddles<Complex>(_size, direction::forward),
    };
};

template<typename Complex, typename Kernel>
fallback_fft_plan<Complex, Kernel>::fallback_fft_plan(fft::order order) : _order{check_order(order)}
{}

template<typename Complex, typename Kernel>
constexpr auto fallback_fft_plan<Complex, Kernel>::max_order() noexcept -> fft::order
{
    return fft::order{27};
}

template<typename Complex, typename Kernel>
constexpr auto fallback_fft_plan<Complex, Kernel>::max_size() noexcept -> size_type
{
    return fft::size(max_order());
}

template<typename Complex, typename Kernel>
auto fallback_fft_plan<Complex, Kernel>::order() const noexcept -> fft::order
{
    return _order;
}

template<typename Complex, typename Kernel>
auto fallback_fft_plan<Complex, Kernel>::size() const noexcept -> size_type
{
    return _size;
}

template<typename Complex, typename Kernel>
template<inout_vector Vec>
    requires std::same_as<typename Vec::value_type, Complex>
auto fallback_fft_plan<Complex, Kernel>::operator()(Vec x, direction dir) noexcept -> void
{
    assert(std::cmp_equal(x.size(), _size));

    _reorder(x);

    if (auto const kernel = Kernel{}; dir == direction::forward) {
        kernel(x, _twiddles.to_mdspan());
    } else {
        kernel(x, conjugate_view{_twiddles.to_mdspan()});
    }
}

template<typename Complex, typename Kernel>
auto fallback_fft_plan<Complex, Kernel>::check_order(fft::order order) -> fft::order
{
    if (order > max_order()) {
        throw std::runtime_error{"fallback: unsupported order '" + std::to_string(int(order)) + "'"};
    }
    return order;
}

}  // namespace neo::fft
