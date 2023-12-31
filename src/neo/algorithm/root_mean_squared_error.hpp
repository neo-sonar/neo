// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/mean_squared_error.hpp>
#include <neo/container/mdspan.hpp>

#include <cassert>
#include <cmath>
#include <concepts>
#include <type_traits>
#include <utility>

namespace neo {

template<in_object InObjL, in_object InObjR>
    requires(InObjL::rank() == InObjR::rank())
[[nodiscard]] auto root_mean_squared_error(InObjL lhs, InObjR rhs) noexcept
{
    return std::sqrt(mean_squared_error(lhs, rhs));
}

}  // namespace neo
