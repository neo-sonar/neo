// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/order.hpp>
#include <neo/type_traits/always_false.hpp>

#include <ipp.h>

#include <cassert>
#include <memory>
#include <stdexcept>

namespace neo::fft {

namespace detail {
struct ipp_free
{
    auto operator()(auto* ptr) const noexcept -> void { ::ippsFree(ptr); }
};

using ipp_buffer = std::unique_ptr<Ipp8u[], ipp_free>;

inline auto ipp_check(int status) -> void
{
    auto const error_msg = [](int ec) -> std::string {
        using namespace std::string_view_literals;

        constexpr auto const errors = std::array{
            std::pair{ ippStsNullPtrErr,    "ipp: internal error"sv},
            std::pair{ippStsFftOrderErr, "ipp: unsupported order"sv},
            std::pair{ ippStsFftFlagErr, "ipp: unsupported flags"sv},
        };

        auto const found = std::ranges::find(errors, ec, &std::pair<int, std::string_view>::first);
        if (found == std::ranges::end(errors)) {
            return "ipp: unkown error" + std::to_string(ec);
        }
        return std::string{found->second};
    };

    if (status != ippStsNoErr) {
        throw std::runtime_error{error_msg(status)};
    }
}

template<typename Setup>
[[nodiscard]] auto make_ipp_fft_handle(std::size_t order)
    -> std::tuple<typename Setup::handle_type*, ipp_buffer, ipp_buffer>
{
    static constexpr auto flag = IPP_FFT_NODIV_BY_ANY;
    static constexpr auto hint = ippAlgHintNone;

    int spec_size = 0;
    int init_size = 0;
    int work_size = 0;
    ipp_check(Setup::get_size(static_cast<int>(order), flag, hint, &spec_size, &init_size, &work_size));

    auto* handle        = static_cast<typename Setup::handle_type*>(nullptr);
    auto spec_buf       = ipp_buffer{::ippsMalloc_8u(spec_size)};
    auto const init_buf = ipp_buffer{::ippsMalloc_8u(init_size)};
    ipp_check(Setup::init(&handle, static_cast<int>(order), flag, hint, spec_buf.get(), init_buf.get()));

    return {handle, std::move(spec_buf), ipp_buffer{::ippsMalloc_8u(work_size)}};
}

template<typename Setup>
[[nodiscard]] auto make_ipp_dft_handle(std::size_t size)
    -> std::tuple<typename Setup::handle_type*, ipp_buffer, ipp_buffer>
{
    static constexpr auto flag = IPP_FFT_NODIV_BY_ANY;
    static constexpr auto hint = ippAlgHintNone;

    auto spec_size = 0;
    auto init_size = 0;
    auto work_size = 0;
    ipp_check(Setup::get_size(static_cast<int>(size), flag, hint, &spec_size, &init_size, &work_size));

    auto const init_buf = ipp_buffer{::ippsMalloc_8u(init_size)};
    auto spec_buf       = ipp_buffer{::ippsMalloc_8u(spec_size)};
    auto* handle        = reinterpret_cast<typename Setup::handle_type*>(spec_buf.get());
    ipp_check(Setup::init(static_cast<int>(size), flag, hint, handle, init_buf.get()));

    return {handle, std::move(spec_buf), ipp_buffer{::ippsMalloc_8u(work_size)}};
}

template<typename Setup>
[[nodiscard]] auto make_ipp_dct_handle(std::size_t order)
    -> std::tuple<typename Setup::handle_type*, ipp_buffer, ipp_buffer>
{
    static constexpr auto hint = ippAlgHintNone;

    auto const len = static_cast<int>(size_t(1) << order);

    int spec_size = 0;
    int init_size = 0;
    int work_size = 0;
    ipp_check(Setup::get_size(len, hint, &spec_size, &init_size, &work_size));

    auto* handle        = static_cast<typename Setup::handle_type*>(nullptr);
    auto spec_buf       = ipp_buffer{::ippsMalloc_8u(spec_size)};
    auto const init_buf = ipp_buffer{::ippsMalloc_8u(init_size)};
    ipp_check(Setup::init(&handle, len, hint, spec_buf.get(), init_buf.get()));

    return {handle, std::move(spec_buf), ipp_buffer{::ippsMalloc_8u(work_size)}};
}

}  // namespace detail

/// \ingroup neo-fft
template<complex Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
struct intel_ipp_fft_plan
{
    using value_type = Complex;
    using real_type  = typename Complex::value_type;
    using size_type  = std::size_t;

    intel_ipp_fft_plan(from_order_tag /*tag*/, size_type order)
        : _order{order}
        , _buffer(stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{size()})
    {
        std::tie(_handle, _spec_buf, _work_buf) = detail::make_ipp_fft_handle<setup>(order);
    }

    intel_ipp_fft_plan(intel_ipp_fft_plan const& other)                    = delete;
    auto operator=(intel_ipp_fft_plan const& other) -> intel_ipp_fft_plan& = delete;

    intel_ipp_fft_plan(intel_ipp_fft_plan&& other)                    = default;
    auto operator=(intel_ipp_fft_plan&& other) -> intel_ipp_fft_plan& = default;

    ~intel_ipp_fft_plan() = default;

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return setup::max_order; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return neo::ipow<2zu>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<inout_vector_of<Complex> InOutVec>
    auto operator()(InOutVec x, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.extent(0), size()));

        auto transform = dir == direction::forward ? setup::forward_inplace : setup::backward_inplace;

        if constexpr (always_vectorizable<InOutVec>) {
            auto buffer = reinterpret_cast<typename setup::complex_type*>(x.data_handle());
            transform(buffer, _handle, _work_buf.get());
        } else {
            copy(x, _buffer.to_mdspan());
            auto buffer = reinterpret_cast<typename setup::complex_type*>(_buffer.data());
            transform(buffer, _handle, _work_buf.get());
            copy(_buffer.to_mdspan(), x);
        }
    }

    template<in_vector_of<Complex> InVec, out_vector_of<Complex> OutVec>
    auto operator()(InVec input, OutVec output, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(input.extent(0), size()));
        assert(std::cmp_equal(output.extent(0), size()));

        if constexpr (always_vectorizable<InVec, OutVec>) {
            auto const* in = reinterpret_cast<typename setup::complex_type const*>(input.data_handle());
            auto* out      = reinterpret_cast<typename setup::complex_type*>(output.data_handle());
            auto transform = dir == direction::forward ? setup::forward_copy : setup::backward_copy;
            transform(in, out, _handle, _work_buf.get());
            return;
        }

        auto buf = _buffer.to_mdspan();
        copy(input, buf);
        (*this)(buf, dir);
        copy(buf, output);
    }

private:
    struct setup_f32
    {
        using complex_type                     = ::Ipp32fc;
        using handle_type                      = ::IppsFFTSpec_C_32fc;
        static constexpr auto max_order        = size_type{28};
        static constexpr auto get_size         = ::ippsFFTGetSize_C_32fc;
        static constexpr auto init             = ::ippsFFTInit_C_32fc;
        static constexpr auto forward_copy     = ::ippsFFTFwd_CToC_32fc;
        static constexpr auto backward_copy    = ::ippsFFTInv_CToC_32fc;
        static constexpr auto forward_inplace  = ::ippsFFTFwd_CToC_32fc_I;
        static constexpr auto backward_inplace = ::ippsFFTInv_CToC_32fc_I;
    };

    struct setup_f64
    {
        using complex_type                     = ::Ipp64fc;
        using handle_type                      = ::IppsFFTSpec_C_64fc;
        static constexpr auto max_order        = size_type{27};
        static constexpr auto get_size         = ::ippsFFTGetSize_C_64fc;
        static constexpr auto init             = ::ippsFFTInit_C_64fc;
        static constexpr auto forward_copy     = ::ippsFFTFwd_CToC_64fc;
        static constexpr auto backward_copy    = ::ippsFFTInv_CToC_64fc;
        static constexpr auto forward_inplace  = ::ippsFFTFwd_CToC_64fc_I;
        static constexpr auto backward_inplace = ::ippsFFTInv_CToC_64fc_I;
    };

    using setup = std::conditional_t<std::same_as<real_type, float>, setup_f32, setup_f64>;

    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<size_t, 1>> _buffer{};
    typename setup::handle_type* _handle;
    detail::ipp_buffer _spec_buf;
    detail::ipp_buffer _work_buf;
};

/// \ingroup neo-fft
template<complex Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
struct intel_ipp_dft_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit intel_ipp_dft_plan(size_type size) : _size{size}
    {
        std::tie(_handle, _spec_buf, _work_buf) = detail::make_ipp_dft_handle<setup>(size);
    }

    intel_ipp_dft_plan(intel_ipp_dft_plan const& other)                    = delete;
    auto operator=(intel_ipp_dft_plan const& other) -> intel_ipp_dft_plan& = delete;

    intel_ipp_dft_plan(intel_ipp_dft_plan&& other)                    = default;
    auto operator=(intel_ipp_dft_plan&& other) -> intel_ipp_dft_plan& = default;

    ~intel_ipp_dft_plan() = default;

    [[nodiscard]] auto size() const noexcept -> size_type { return _size; }

    template<inout_vector_of<Complex> InOutVec>
    auto operator()(InOutVec x, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.extent(0), size()));

        copy(x, _tmp_in.to_mdspan());
        (*this)(_tmp_in.to_mdspan(), _tmp_out.to_mdspan(), dir);
        copy(_tmp_out.to_mdspan(), x);
    }

    template<in_vector_of<Complex> InVec, out_vector_of<Complex> OutVec>
    auto operator()(InVec input, OutVec output, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(input.extent(0), size()));
        assert(std::cmp_equal(output.extent(0), size()));

        auto run = [this, dir](auto const* in_ptr, auto* out_ptr) {
            auto const* in = reinterpret_cast<typename setup::complex_type const*>(in_ptr);
            auto* out      = reinterpret_cast<typename setup::complex_type*>(out_ptr);
            auto transform = dir == direction::forward ? setup::forward : setup::backward;
            transform(in, out, _handle, _work_buf.get());
        };

        if constexpr (always_vectorizable<InVec, OutVec>) {
            run(input.data_handle(), output.data_handle());
        } else {
            copy(input, _tmp_in.to_mdspan());
            run(_tmp_in.data(), _tmp_out.data());
            copy(_tmp_out.to_mdspan(), output);
        }
    }

private:
    struct setup_f32
    {
        using complex_type             = ::Ipp32fc;
        using handle_type              = ::IppsDFTSpec_C_32fc;
        static constexpr auto get_size = ::ippsDFTGetSize_C_32fc;
        static constexpr auto init     = ::ippsDFTInit_C_32fc;
        static constexpr auto forward  = ::ippsDFTFwd_CToC_32fc;
        static constexpr auto backward = ::ippsDFTInv_CToC_32fc;
    };

    struct setup_f64
    {
        using complex_type             = ::Ipp64fc;
        using handle_type              = ::IppsDFTSpec_C_64fc;
        static constexpr auto get_size = ::ippsDFTGetSize_C_64fc;
        static constexpr auto init     = ::ippsDFTInit_C_64fc;
        static constexpr auto forward  = ::ippsDFTFwd_CToC_64fc;
        static constexpr auto backward = ::ippsDFTInv_CToC_64fc;
    };

    using setup = std::conditional_t<std::same_as<typename Complex::value_type, float>, setup_f32, setup_f64>;

    size_type _size;
    stdex::mdarray<Complex, stdex::dextents<size_t, 1>> _tmp_in{size()};
    stdex::mdarray<Complex, stdex::dextents<size_t, 1>> _tmp_out{size()};
    typename setup::handle_type* _handle;
    detail::ipp_buffer _spec_buf;
    detail::ipp_buffer _work_buf;
};

/// \ingroup neo-fft
template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
struct intel_ipp_split_fft_plan
{
    using value_type = Float;
    using size_type  = std::size_t;

    intel_ipp_split_fft_plan(from_order_tag /*tag*/, size_type order) : _order{order}
    {
        std::tie(_handle, _spec_buf, _work_buf) = detail::make_ipp_fft_handle<setup>(order);
    }

    intel_ipp_split_fft_plan(intel_ipp_split_fft_plan const& other)                    = delete;
    auto operator=(intel_ipp_split_fft_plan const& other) -> intel_ipp_split_fft_plan& = delete;

    intel_ipp_split_fft_plan(intel_ipp_split_fft_plan&& other)                    = default;
    auto operator=(intel_ipp_split_fft_plan&& other) -> intel_ipp_split_fft_plan& = default;

    ~intel_ipp_split_fft_plan() = default;

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<inout_vector_of<Float> InOutVec>
    auto operator()(split_complex<InOutVec> x, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.real.extent(0), size()));
        assert(neo::detail::extents_equal(x.real, x.imag));

        auto transform = dir == direction::forward ? setup::forward_inplace : setup::backward_inplace;

        if constexpr (always_vectorizable<InOutVec>) {
            transform(x.real.data_handle(), x.imag.data_handle(), _handle, _work_buf.get());
        } else {
            always_false<InOutVec>;
        }
    }

    template<in_vector_of<Float> InVec, out_vector_of<Float> OutVec>
    auto operator()(split_complex<InVec> in, split_complex<OutVec> out, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(in.real.extent(0), size()));
        assert(neo::detail::extents_equal(in.real, in.imag, out.real, out.imag));

        auto transform = dir == direction::forward ? setup::forward_copy : setup::backward_copy;

        if constexpr (always_vectorizable<InVec> and always_vectorizable<OutVec>) {
            transform(
                in.real.data_handle(),
                in.imag.data_handle(),
                out.real.data_handle(),
                out.imag.data_handle(),
                _handle,
                _work_buf.get()
            );
        } else {
            always_false<InVec>;
        }
    }

private:
    struct setup_f32
    {
        using handle_type                      = ::IppsFFTSpec_C_32f;
        static constexpr auto get_size         = ::ippsFFTGetSize_C_32f;
        static constexpr auto init             = ::ippsFFTInit_C_32f;
        static constexpr auto forward_copy     = ::ippsFFTFwd_CToC_32f;
        static constexpr auto backward_copy    = ::ippsFFTInv_CToC_32f;
        static constexpr auto forward_inplace  = ::ippsFFTFwd_CToC_32f_I;
        static constexpr auto backward_inplace = ::ippsFFTInv_CToC_32f_I;
    };

    struct setup_f64
    {
        using handle_type                      = ::IppsFFTSpec_C_64f;
        static constexpr auto get_size         = ::ippsFFTGetSize_C_64f;
        static constexpr auto init             = ::ippsFFTInit_C_64f;
        static constexpr auto forward_copy     = ::ippsFFTFwd_CToC_64f;
        static constexpr auto backward_copy    = ::ippsFFTInv_CToC_64f;
        static constexpr auto forward_inplace  = ::ippsFFTFwd_CToC_64f_I;
        static constexpr auto backward_inplace = ::ippsFFTInv_CToC_64f_I;
    };

    using setup = std::conditional_t<std::same_as<Float, float>, setup_f32, setup_f64>;

    size_type _order;
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _buffer{2, size()};
    typename setup::handle_type* _handle;
    detail::ipp_buffer _spec_buf;
    detail::ipp_buffer _work_buf;
};

/// \ingroup neo-fft
template<std::floating_point Float, complex Complex = std::complex<Float>>
    requires(
        (std::same_as<Float, float> or std::same_as<Float, double>)
        and std::same_as<typename Complex::value_type, Float>
    )
struct intel_ipp_rfft_plan
{
    using real_type    = Float;
    using complex_type = Complex;
    using size_type    = std::size_t;

    intel_ipp_rfft_plan(from_order_tag /*tag*/, size_type order) : _order{order}
    {
        std::tie(_handle, _spec_buf, _work_buf) = detail::make_ipp_fft_handle<setup>(order);
    }

    intel_ipp_rfft_plan(intel_ipp_rfft_plan const& other)                    = delete;
    auto operator=(intel_ipp_rfft_plan const& other) -> intel_ipp_rfft_plan& = delete;

    intel_ipp_rfft_plan(intel_ipp_rfft_plan&& other)                    = default;
    auto operator=(intel_ipp_rfft_plan&& other) -> intel_ipp_rfft_plan& = default;

    ~intel_ipp_rfft_plan() = default;

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<in_vector_of<Float> InVec, out_vector_of<complex_type> OutVec>
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        assert(std::cmp_equal(in.extent(0), size()));

        if constexpr (always_vectorizable<InVec> and always_vectorizable<OutVec>
                      and (sizeof(complex_type) == sizeof(Float) * 2)) {
            auto* const out_ptr = reinterpret_cast<Float*>(out.data_handle());
            setup::forward_copy(in.data_handle(), out_ptr, _handle, _work_buf.get());
        } else {
            auto buf = _buffer.to_mdspan();
            copy(in, stdex::submdspan(buf, std::tuple{0, size()}));

            setup::forward_inplace(_buffer.data(), _handle, _work_buf.get());

            auto const coeffs = size() / 2 + 1;
            for (auto i{0U}; i < coeffs; ++i) {
                out[i] = complex_type{buf[i * 2], buf[i * 2 + 1]};
            }
        }
    }

    template<in_vector_of<complex_type> InVec, out_vector_of<Float> OutVec>
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        if constexpr (always_vectorizable<InVec> and always_vectorizable<OutVec>
                      and (sizeof(complex_type) == sizeof(Float) * 2)) {
            auto* const in_ptr = reinterpret_cast<Float*>(in.data_handle());
            setup::backward_copy(in_ptr, out.data_handle(), _handle, _work_buf.get());
        } else {
            auto buf = _buffer.to_mdspan();
            for (auto i{0U}; i < in.size(); ++i) {
                buf[i * 2]     = real(in[i]);
                buf[i * 2 + 1] = imag(in[i]);
            }

            setup::backward_inplace(_buffer.data(), _handle, _work_buf.get());
            copy(stdex::submdspan(buf, std::tuple{0, size()}), out);
        }
    }

private:
    struct setup_f32
    {
        using float_type                       = ::Ipp32f;
        using handle_type                      = ::IppsFFTSpec_R_32f;
        static constexpr auto get_size         = ::ippsFFTGetSize_R_32f;
        static constexpr auto init             = ::ippsFFTInit_R_32f;
        static constexpr auto forward_copy     = ::ippsFFTFwd_RToCCS_32f;
        static constexpr auto backward_copy    = ::ippsFFTInv_CCSToR_32f;
        static constexpr auto forward_inplace  = ::ippsFFTFwd_RToCCS_32f_I;
        static constexpr auto backward_inplace = ::ippsFFTInv_CCSToR_32f_I;
    };

    struct setup_f64
    {
        using float_type                       = ::Ipp64f;
        using handle_type                      = ::IppsFFTSpec_R_64f;
        static constexpr auto get_size         = ::ippsFFTGetSize_R_64f;
        static constexpr auto init             = ::ippsFFTInit_R_64f;
        static constexpr auto forward_copy     = ::ippsFFTFwd_RToCCS_64f;
        static constexpr auto backward_copy    = ::ippsFFTInv_CCSToR_64f;
        static constexpr auto forward_inplace  = ::ippsFFTFwd_RToCCS_64f_I;
        static constexpr auto backward_inplace = ::ippsFFTInv_CCSToR_64f_I;
    };

    using setup = std::conditional_t<std::same_as<real_type, float>, setup_f32, setup_f64>;

    size_type _order;
    stdex::mdarray<typename setup::float_type, stdex::dextents<size_t, 1>> _buffer{size() * 2};
    typename setup::handle_type* _handle;
    detail::ipp_buffer _spec_buf;
    detail::ipp_buffer _work_buf;
};

/// \ingroup neo-fft
template<std::floating_point Float, direction Direction>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
struct intel_ipp_dct_plan
{
    using value_type = Float;
    using size_type  = std::size_t;

    intel_ipp_dct_plan(from_order_tag /*tag*/, size_type order) : _order{order}
    {
        std::tie(_handle, _spec_buf, _work_buf) = detail::make_ipp_dct_handle<setup>(order);
    }

    intel_ipp_dct_plan(intel_ipp_dct_plan const& other)                    = delete;
    auto operator=(intel_ipp_dct_plan const& other) -> intel_ipp_dct_plan& = delete;

    intel_ipp_dct_plan(intel_ipp_dct_plan&& other)                    = default;
    auto operator=(intel_ipp_dct_plan&& other) -> intel_ipp_dct_plan& = default;

    ~intel_ipp_dct_plan() = default;

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Float>
    auto operator()(Vec x) noexcept -> void
    {
        auto const buf = _buffer.to_mdspan();
        copy(x, buf);
        setup::transform_inplace(_buffer.data(), _handle, _work_buf.get());
        copy(buf, x);
    }

private:
    struct dct2_setup_f32
    {
        using value_type                        = ::Ipp32f;
        using handle_type                       = ::IppsDCTFwdSpec_32f;
        static constexpr auto get_size          = ::ippsDCTFwdGetSize_32f;
        static constexpr auto init              = ::ippsDCTFwdInit_32f;
        static constexpr auto transform_copy    = ::ippsDCTFwd_32f;
        static constexpr auto transform_inplace = ::ippsDCTFwd_32f_I;
    };

    struct dct2_setup_f64
    {
        using value_type                        = ::Ipp64f;
        using handle_type                       = ::IppsDCTFwdSpec_64f;
        static constexpr auto get_size          = ::ippsDCTFwdGetSize_64f;
        static constexpr auto init              = ::ippsDCTFwdInit_64f;
        static constexpr auto transform_copy    = ::ippsDCTFwd_64f;
        static constexpr auto transform_inplace = ::ippsDCTFwd_64f_I;
    };

    struct dct3_setup_f32
    {
        using value_type                        = ::Ipp32f;
        using handle_type                       = ::IppsDCTInvSpec_32f;
        static constexpr auto get_size          = ::ippsDCTInvGetSize_32f;
        static constexpr auto init              = ::ippsDCTInvInit_32f;
        static constexpr auto transform_copy    = ::ippsDCTInv_32f;
        static constexpr auto transform_inplace = ::ippsDCTInv_32f_I;
    };

    struct dct3_setup_f64
    {
        using value_type                        = ::Ipp64f;
        using handle_type                       = ::IppsDCTInvSpec_64f;
        static constexpr auto get_size          = ::ippsDCTInvGetSize_64f;
        static constexpr auto init              = ::ippsDCTInvInit_64f;
        static constexpr auto transform_copy    = ::ippsDCTInv_64f;
        static constexpr auto transform_inplace = ::ippsDCTInv_64f_I;
    };

    using dct2_setup = std::conditional_t<std::same_as<Float, float>, dct2_setup_f32, dct2_setup_f64>;
    using dct3_setup = std::conditional_t<std::same_as<Float, float>, dct3_setup_f32, dct3_setup_f64>;
    using setup      = std::conditional_t<Direction == direction::forward, dct2_setup, dct3_setup>;

    size_type _order;
    stdex::mdarray<typename setup::value_type, stdex::dextents<size_t, 1>> _buffer{size()};
    typename setup::handle_type* _handle;
    detail::ipp_buffer _spec_buf;
    detail::ipp_buffer _work_buf;
};

template<std::floating_point Float>
using intel_ipp_dct2_plan = intel_ipp_dct_plan<Float, direction::forward>;

template<std::floating_point Float>
using intel_ipp_dct3_plan = intel_ipp_dct_plan<Float, direction::backward>;

}  // namespace neo::fft
