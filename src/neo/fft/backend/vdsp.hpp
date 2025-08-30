// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/order.hpp>
#include <neo/type_traits/always_false.hpp>

#include <Accelerate/Accelerate.h>

#include <cassert>
#include <utility>

namespace neo::fft {

/// \ingroup neo-fft
template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
struct apple_vdsp_fft_plan
{
    using value_type         = Complex;
    using real_type          = typename Complex::value_type;
    using size_type          = std::size_t;
    using native_handle_type = std::conditional_t<std::same_as<real_type, float>, FFTSetup, FFTSetupD>;

    apple_vdsp_fft_plan(from_order_tag /*tag*/, size_type order);
    ~apple_vdsp_fft_plan();

    apple_vdsp_fft_plan(apple_vdsp_fft_plan const& other)                    = delete;
    auto operator=(apple_vdsp_fft_plan const& other) -> apple_vdsp_fft_plan& = delete;

    apple_vdsp_fft_plan(apple_vdsp_fft_plan&& other) noexcept
        : _order{std::exchange(other._order, size_type{0})}
        , _size{std::exchange(other._size, 0)}
        , _plan{std::exchange(other._plan, nullptr)}
        , _input{std::move(other._input)}
        , _output{std::move(other._output)}
    {}

    auto operator=(apple_vdsp_fft_plan&& other) noexcept -> apple_vdsp_fft_plan&
    {
        _order  = std::exchange(other._order, size_type{0});
        _size   = std::exchange(other._size, 0);
        _plan   = std::exchange(other._plan, nullptr);
        _input  = std::move(other._input);
        _output = std::move(other._output);
        return *this;
    }

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{27}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return neo::ipow<2zu>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type;
    [[nodiscard]] auto size() const noexcept -> size_type;

    template<inout_vector_of<Complex> InOutVec>
    auto operator()(InOutVec x, direction dir) noexcept -> void;

private:
    size_type _order;
    size_type _size{neo::ipow<2zu>(order())};
    native_handle_type _plan;
    stdex::mdarray<real_type, stdex::dextents<size_t, 2>> _input{2, _size};
    stdex::mdarray<real_type, stdex::dextents<size_t, 2>> _output{2, _size};
};

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
apple_vdsp_fft_plan<Complex>::apple_vdsp_fft_plan(from_order_tag /*tag*/, size_type order)
    : _order{order}
    , _plan{[order] {
        if (order > max_order()) {
            throw std::runtime_error{"vdsp: unsupported order '" + std::to_string(int(order)) + "'"};
        }

        if constexpr (std::same_as<real_type, float>) {
            return vDSP_create_fftsetup(static_cast<vDSP_Length>(order), 2);
        } else {
            return vDSP_create_fftsetupD(static_cast<vDSP_Length>(order), 2);
        }
    }()}
{
    assert(_plan != nullptr);
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
apple_vdsp_fft_plan<Complex>::~apple_vdsp_fft_plan()
{
    if (_plan != nullptr) {
        if constexpr (std::same_as<real_type, float>) {
            vDSP_destroy_fftsetup(_plan);
        } else {
            vDSP_destroy_fftsetupD(_plan);
        }
    }
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
auto apple_vdsp_fft_plan<Complex>::size() const noexcept -> size_type
{
    return _size;
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
auto apple_vdsp_fft_plan<Complex>::order() const noexcept -> size_type
{
    return _order;
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
template<inout_vector_of<Complex> InOutVec>
auto apple_vdsp_fft_plan<Complex>::operator()(InOutVec x, direction dir) noexcept -> void
{
    assert(std::cmp_equal(x.extent(0), _size));

    using split_complex = std::conditional_t<std::same_as<real_type, float>, DSPSplitComplex, DSPDoubleSplitComplex>;

    auto const in   = split_complex{.realp = std::addressof(_input(0, 0)), .imagp = std::addressof(_input(1, 0))};
    auto const out  = split_complex{.realp = std::addressof(_output(0, 0)), .imagp = std::addressof(_output(1, 0))};
    auto const sign = dir == direction::forward ? kFFTDirection_Forward : kFFTDirection_Inverse;

    for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
        in.realp[i] = x[i].real();
        in.imagp[i] = x[i].imag();
    }

    if constexpr (std::same_as<real_type, float>) {
        vDSP_fft_zop(_plan, &in, 1, &out, 1, static_cast<vDSP_Length>(_order), sign);
    } else {
        vDSP_fft_zopD(_plan, &in, 1, &out, 1, static_cast<vDSP_Length>(_order), sign);
    }

    for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
        x[i] = Complex{out.realp[i], out.imagp[i]};
    }
}

/// \ingroup neo-fft
template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
struct apple_vdsp_split_fft_plan
{
    using value_type         = Float;
    using size_type          = std::size_t;
    using native_handle_type = std::conditional_t<std::same_as<Float, float>, FFTSetup, FFTSetupD>;

    apple_vdsp_split_fft_plan(from_order_tag /*tag*/, size_type order);
    ~apple_vdsp_split_fft_plan();

    apple_vdsp_split_fft_plan(apple_vdsp_split_fft_plan const& other)                    = delete;
    auto operator=(apple_vdsp_split_fft_plan const& other) -> apple_vdsp_split_fft_plan& = delete;

    apple_vdsp_split_fft_plan(apple_vdsp_split_fft_plan&& other) noexcept
        : _order{std::exchange(other._order, size_type{0})}
        , _size{std::exchange(other._size, 0)}
        , _plan{std::exchange(other._plan, nullptr)}
        , _input{std::move(other._input)}
        , _output{std::move(other._output)}
    {}

    auto operator=(apple_vdsp_split_fft_plan&& other) noexcept -> apple_vdsp_split_fft_plan&
    {
        _order  = std::exchange(other._order, size_type{0});
        _size   = std::exchange(other._size, 0);
        _plan   = std::exchange(other._plan, nullptr);
        _input  = std::move(other._input);
        _output = std::move(other._output);
        return *this;
    }

    [[nodiscard]] auto order() const noexcept -> size_type;
    [[nodiscard]] auto size() const noexcept -> size_type;

    template<inout_vector_of<Float> InOutVec>
    auto operator()(split_complex<InOutVec> x, direction dir) noexcept -> void;

    template<in_vector_of<Float> InVec, out_vector_of<Float> OutVec>
    auto operator()(split_complex<InVec> in, split_complex<OutVec> out, direction dir) noexcept -> void;

private:
    size_type _order;
    size_type _size{neo::ipow<2zu>(order())};
    native_handle_type _plan;
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _input{2, _size};
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _output{2, _size};
};

template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
apple_vdsp_split_fft_plan<Float>::apple_vdsp_split_fft_plan(from_order_tag /*tag*/, size_type order)
    : _order{order}
    , _plan{[order] {
        if constexpr (std::same_as<Float, float>) {
            return vDSP_create_fftsetup(static_cast<vDSP_Length>(order), 2);
        } else {
            return vDSP_create_fftsetupD(static_cast<vDSP_Length>(order), 2);
        }
    }()}
{
    assert(_plan != nullptr);
}

template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
apple_vdsp_split_fft_plan<Float>::~apple_vdsp_split_fft_plan()
{
    if (_plan != nullptr) {
        if constexpr (std::same_as<Float, float>) {
            vDSP_destroy_fftsetup(_plan);
        } else {
            vDSP_destroy_fftsetupD(_plan);
        }
    }
}

template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
auto apple_vdsp_split_fft_plan<Float>::size() const noexcept -> size_type
{
    return _size;
}

template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
auto apple_vdsp_split_fft_plan<Float>::order() const noexcept -> size_type
{
    return _order;
}

template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
template<inout_vector_of<Float> InOutVec>
auto apple_vdsp_split_fft_plan<Float>::operator()(split_complex<InOutVec> x, direction dir) noexcept -> void
{
    assert(std::cmp_equal(x.real.extent(0), size()));
    assert(neo::detail::extents_equal(x.real, x.imag));

    using split_complex = std::conditional_t<std::same_as<Float, float>, DSPSplitComplex, DSPDoubleSplitComplex>;

    auto const sign = dir == direction::forward ? kFFTDirection_Forward : kFFTDirection_Inverse;

    if constexpr (always_vectorizable<InOutVec>) {
        auto const split_x = split_complex{
            .realp = x.real.data_handle(),
            .imagp = x.imag.data_handle(),
        };

        if constexpr (std::same_as<Float, float>) {
            vDSP_fft_zip(_plan, &split_x, 1, static_cast<vDSP_Length>(_order), sign);
        } else {
            vDSP_fft_zipD(_plan, &split_x, 1, static_cast<vDSP_Length>(_order), sign);
        }
    } else {
        always_false<InOutVec>;
    }
}

template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
template<in_vector_of<Float> InVec, out_vector_of<Float> OutVec>
auto apple_vdsp_split_fft_plan<Float>::operator()(
    split_complex<InVec> in,
    split_complex<OutVec> out,
    direction dir
) noexcept -> void
{
    assert(std::cmp_equal(in.real.extent(0), size()));
    assert(neo::detail::extents_equal(in.real, in.imag, out.real, out.imag));

    using split_complex = std::conditional_t<std::same_as<Float, float>, DSPSplitComplex, DSPDoubleSplitComplex>;

    auto const sign = dir == direction::forward ? kFFTDirection_Forward : kFFTDirection_Inverse;

    if constexpr (always_vectorizable<InVec, OutVec>) {
        auto const split_in  = split_complex{.realp = in.real.data_handle(), .imagp = in.imag.data_handle()};
        auto const split_out = split_complex{.realp = out.real.data_handle(), .imagp = out.imag.data_handle()};

        if constexpr (std::same_as<Float, float>) {
            vDSP_fft_zop(_plan, &split_in, 1, &split_out, 1, static_cast<vDSP_Length>(_order), sign);
        } else {
            vDSP_fft_zopD(_plan, &split_in, 1, &split_out, 1, static_cast<vDSP_Length>(_order), sign);
        }
    } else {
        always_false<InVec>;
    }
}

}  // namespace neo::fft
