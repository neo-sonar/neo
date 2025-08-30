// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>

#include <mkl_dfti.h>

namespace neo::fft {

/// \ingroup neo-fft
template<complex Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
struct intel_mkl_fft_plan
{
    using value_type = Complex;
    using real_type  = typename Complex::value_type;
    using size_type  = std::size_t;

    intel_mkl_fft_plan(from_order_tag /*tag*/, size_type order) : _order{order}, _handle{make(order)} {}

    ~intel_mkl_fft_plan() noexcept
    {
        if (_handle) {
            DftiFreeDescriptor(&_handle->ptr);
        }
    }

    intel_mkl_fft_plan(intel_mkl_fft_plan const& other)                    = delete;
    auto operator=(intel_mkl_fft_plan const& other) -> intel_mkl_fft_plan& = delete;

    intel_mkl_fft_plan(intel_mkl_fft_plan&& other)                    = default;
    auto operator=(intel_mkl_fft_plan&& other) -> intel_mkl_fft_plan& = default;

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{27}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return neo::ipow<2zu>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return neo::ipow<2zu>(order()); }

    template<inout_vector InOutVec>
        requires std::same_as<typename InOutVec::value_type, Complex>
    auto operator()(InOutVec x, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.extent(0), size()));

        auto perform = [this, dir](auto* ptr) {
            if (dir == direction::forward) {
                DftiComputeForward(_handle->ptr, static_cast<void*>(ptr));
            } else {
                DftiComputeBackward(_handle->ptr, static_cast<void*>(ptr));
            }
        };

        if constexpr (always_vectorizable<InOutVec>) {
            perform(x.data_handle());
        } else {
            copy(x, _buffer.to_mdspan());
            perform(_buffer.data());
            copy(_buffer.to_mdspan(), x);
        }
    }

private:
    struct handle_t
    {
        DFTI_DESCRIPTOR_HANDLE ptr;
    };

    [[nodiscard]] static auto make(size_type order)
    {
        if (order > max_order()) {
            throw std::runtime_error{"mkl: unsupported order '" + std::to_string(int(order)) + "'"};
        }

        static constexpr auto const precision  = std::same_as<real_type, float> ? DFTI_SINGLE : DFTI_DOUBLE;
        static constexpr auto const domain     = DFTI_COMPLEX;
        static constexpr auto const dimensions = 1;

        auto* handle   = DFTI_DESCRIPTOR_HANDLE{};
        auto const len = ipow<size_type(2)>(order);

        DftiCreateDescriptor(&handle, precision, domain, dimensions, len);
        DftiSetValue(handle, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiCommitDescriptor(handle);

        return std::make_unique<handle_t>(handle_t{.ptr = handle});
    }

    size_type _order;
    std::unique_ptr<handle_t> _handle;
    stdex::mdarray<Complex, stdex::dextents<size_t, 1>> _buffer{size()};
};

}  // namespace neo::fft
