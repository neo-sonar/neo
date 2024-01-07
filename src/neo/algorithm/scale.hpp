// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/backend/cblas.hpp>
#include <neo/algorithm/backend/linalg_unary_op.hpp>

namespace neo {

template<typename Scalar, inout_object InOutObj>
constexpr auto scale(Scalar alpha, InOutObj obj) -> void
{
#if defined(NEO_HAS_CBLAS)
    if constexpr (always_vectorizable<InOutObj>) {
        auto const n      = obj.extent(0);
        auto const stride = obj.stride(0);
        auto* const ptr   = obj.data_handle();
        if constexpr (requires { cblas::scal(n, alpha, ptr, stride); }) {
            return cblas::scal(n, alpha, ptr, stride);
        }
    }
#endif
    detail::linalg_unary_op(obj, [alpha](auto const& val) { return val * alpha; });
}

}  // namespace neo
