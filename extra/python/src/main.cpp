// SPDX-License-Identifier: MIT

#include <neo/algorithm.hpp>
#include <neo/complex.hpp>
#include <neo/convolution.hpp>
#include <neo/fft.hpp>
#include <neo/math.hpp>
#include <neo/type_traits.hpp>
#include <neo/unit.hpp>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <bit>
#include <complex>
#include <cstdio>
#include <iostream>
#include <optional>
#include <typeindex>

namespace py = pybind11;

template<typename T, int Flags>
[[nodiscard]] auto is_layout_left(py::array_t<T, Flags> buf) -> bool
{
    return (buf.flags() & py::array::f_style) != 0;
}

template<typename T, int Flags>
[[nodiscard]] auto is_layout_right(py::array_t<T, Flags> buf) -> bool
{
    return (buf.flags() & py::array::c_style) != 0;
}

template<typename T, int Flags, typename Mapping>
[[nodiscard]] auto strides_match(py::array_t<T, Flags> buf, Mapping map) -> bool
{
    for (auto i{0zu}; i < map.extents().rank(); ++i) {
        if (map.stride(i) != static_cast<size_t>(buf.strides(i) / buf.itemsize())) {
            return false;
        }
    }
    return true;
}

template<size_t Dim, typename T, int Flags>
[[nodiscard]] auto make_extents(py::array_t<T, Flags> buf) -> std::array<size_t, Dim>
{
    auto ext = std::array<size_t, Dim>{};
    for (auto i{0zu}; i < Dim; ++i) {
        ext[i] = static_cast<size_t>(buf.shape(i));
    }
    return ext;
}

template<size_t Dim, typename T, int Flags>
[[nodiscard]] auto make_strides(py::array_t<T, Flags> buf) -> std::array<size_t, Dim>
{
    auto strides = std::array<size_t, Dim>{};
    for (auto i{0zu}; i < Dim; ++i) {
        strides[i] = static_cast<size_t>(buf.strides(i) / buf.itemsize());
    }
    return strides;
}

template<std::size_t Dim, typename T, int Flags>
auto to_mdspan_layout_right(py::array_t<T, Flags> buf)
{
    auto const extents = stdex::dextents<size_t, Dim>{make_extents<Dim>(buf)};
    auto const mapping = stdex::layout_right::mapping<stdex::dextents<size_t, Dim>>{extents};
    return stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_right>{
        buf.mutable_data(),
        mapping,
    };
}

template<std::size_t Dim, typename T, int Flags>
auto to_mdspan_layout_stride(py::array_t<T, Flags> buf)
{
    auto const strides = make_strides<Dim>(buf);
    auto const extents = stdex::dextents<size_t, Dim>{make_extents<Dim>(buf)};
    auto const mapping = stdex::layout_stride::mapping<stdex::dextents<size_t, Dim>>{extents, strides};
    return stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_stride>{
        buf.mutable_data(),
        mapping,
    };
}

template<std::size_t Dim, typename T, int Flags>
auto as_mdspan_impl(py::array_t<T, Flags> buf, auto func)
{
    auto const extents = stdex::dextents<size_t, Dim>{make_extents<Dim>(buf)};

    auto const mapping_right = stdex::layout_right::mapping<stdex::dextents<size_t, Dim>>{extents};
    if (is_layout_right(buf) and strides_match(buf, mapping_right)) {
        return func(
            stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_right>{
                buf.mutable_data(),
                mapping_right,
            }
        );
    }

    auto const mapping_left = stdex::layout_left::mapping<stdex::dextents<size_t, Dim>>{extents};
    if (is_layout_left(buf) and strides_match(buf, mapping_left)) {
        return func(
            stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_left>{
                buf.mutable_data(),
                mapping_left,
            }
        );
    }

    return func(to_mdspan_layout_stride<Dim>(buf));
}

template<int MaxDim, typename T, int Flags>
auto as_mdspan(py::array_t<T, Flags> buf, auto func)
{
    auto const dim = buf.ndim();

    if constexpr (MaxDim >= 1) {
        if (dim == 1) {
            return as_mdspan_impl<1>(buf, func);
        }
    }

    throw std::runtime_error("unsupported ndim: " + std::to_string(dim));
}

template<neo::complex Complex, neo::fft::direction Dir>
auto fft(py::array_t<Complex> array, std::optional<std::size_t> n, neo::fft::norm norm) -> py::array_t<Complex>
{
    using Float = neo::value_type_t<Complex>;

    return as_mdspan<1>(array, [n, norm](neo::in_vector auto input) -> py::array_t<Complex> {
        auto const size  = n.value_or(input.extent(0));
        auto const order = neo::fft::next_order(size);
        if (not std::has_single_bit(size)) {
            throw std::runtime_error{"unsupported size: " + std::to_string(size)};
        }

        auto result = py::array_t<Complex>(static_cast<py::ssize_t>(size));
        auto out    = to_mdspan_layout_right<1>(result);

        {
            auto no_gil = py::gil_scoped_release{};

            auto plan = neo::fft::fft_plan<Complex>{neo::fft::from_order, order};
            if constexpr (Dir == neo::fft::direction::forward) {
                neo::fft::fft(plan, input, out);
                if (norm == neo::fft::norm::forward) {
                    neo::scale(Float(1) / Float(size), out);
                }
            } else {
                neo::fft::ifft(plan, input, out);
                if (norm == neo::fft::norm::backward) {
                    neo::scale(Float(1) / Float(size), out);
                }
            }

            if (norm == neo::fft::norm::ortho) {
                neo::scale(Float(1) / std::sqrt(Float(size)), out);
            }
        }

        return result;
    });
}

template<neo::convolution::method Method, std::floating_point Float>
[[nodiscard]] auto convolve(py::array_t<Float> in1, py::array_t<Float> in2, neo::convolution::mode mode)
    -> py::array_t<Float>
{
    if (in1.ndim() != 1 or in1.ndim() != 1) {
        throw std::runtime_error{"unsupported dimension: in1 and in2 must be 1-D"};
    }

    auto const signal = to_mdspan_layout_stride<1>(in1);
    auto const patch  = to_mdspan_layout_stride<1>(in2);

    if (mode == neo::convolution::mode::full) {
        auto output      = py::array_t<Float>(static_cast<py::ssize_t>(signal.extent(0) + patch.extent(0) - 1));
        auto output_view = to_mdspan_layout_right<1>(output);

        {
            auto no_gil = py::gil_scoped_release{};
            if constexpr (Method == neo::convolution::method::direct) {
                neo::convolution::direct_convolve(signal, patch, output_view);
            } else if constexpr (Method == neo::convolution::method::fft) {
                auto out = neo::convolution::fft_convolve(signal, patch);
                neo::copy(out.to_mdspan(), output_view);
            }
        }

        return output;
    }

    throw std::runtime_error{"unsupported convolution mode"};
}

template<std::floating_point Float>
[[nodiscard]] auto amplitude_to_db(Float amplitude) -> Float
{
    return neo::amplitude_to_db<neo::precision::accurate, Float>(amplitude);
}

template<std::floating_point Float>
[[nodiscard]] auto amplitude_to_db_estimate(Float amplitude) -> Float
{
    return neo::amplitude_to_db<neo::precision::estimate, Float>(amplitude);
}

[[nodiscard]] auto rfftfreq(std::size_t n, double invSampleRate) -> py::array_t<double>
{
    auto out  = py::array_t<double>(static_cast<py::ssize_t>(n));
    auto view = to_mdspan_layout_right<1>(out);

    {
        auto no_gil = py::gil_scoped_release{};
        neo::rfftfreq(view, invSampleRate);
    }

    return out;
}

PYBIND11_MODULE(_neo, m)
{
    py::enum_<neo::convolution::method>(m, "convolution_method")
        .value("automatic", neo::convolution::method::automatic)
        .value("direct", neo::convolution::method::direct)
        .value("fft", neo::convolution::method::fft)
        .value("ola", neo::convolution::method::ola)
        .value("ols", neo::convolution::method::ols)
        .value("upola", neo::convolution::method::upola)
        .value("upols", neo::convolution::method::upols);

    py::enum_<neo::convolution::mode>(m, "convolution_mode")
        .value("full", neo::convolution::mode::full)
        .value("valid", neo::convolution::mode::valid)
        .value("same", neo::convolution::mode::same);

    py::enum_<neo::fft::norm>(m, "norm")
        .value("backward", neo::fft::norm::backward)
        .value("ortho", neo::fft::norm::ortho)
        .value("forward", neo::fft::norm::forward);

    m.def("rfftfreq", &rfftfreq);

    m.def("fft", &fft<std::complex<float>, neo::fft::direction::forward>);
    m.def("fft", &fft<std::complex<double>, neo::fft::direction::forward>);

    m.def("ifft", &fft<std::complex<float>, neo::fft::direction::backward>);
    m.def("ifft", &fft<std::complex<double>, neo::fft::direction::backward>);

    m.def("direct_convolve", &convolve<neo::convolution::method::direct, float>);
    m.def("direct_convolve", &convolve<neo::convolution::method::direct, double>);

    m.def("fft_convolve", &convolve<neo::convolution::method::fft, float>);
    m.def("fft_convolve", &convolve<neo::convolution::method::fft, double>);

    m.def("amplitude_to_db", py::vectorize(amplitude_to_db<float>));
    m.def("amplitude_to_db", py::vectorize(amplitude_to_db<double>));

    m.def("amplitude_to_db_estimate", py::vectorize(amplitude_to_db_estimate<float>));
    m.def("amplitude_to_db_estimate", py::vectorize(amplitude_to_db_estimate<double>));

    m.def("a_weighting", py::vectorize(neo::a_weighting<float>));
    m.def("a_weighting", py::vectorize(neo::a_weighting<double>));

    m.def("fast_log2", py::vectorize(neo::fast_log2));
    m.def("fast_log10", py::vectorize(neo::fast_log10));

#ifdef VERSION_INFO
    m.attr("__version__") = NEO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
