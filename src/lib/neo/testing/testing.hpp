#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/fill.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>

#include <algorithm>
#include <concepts>
#include <random>
#include <vector>

namespace neo {

namespace detail {

template<typename T>
consteval auto real_or_complex_value()
{
    if constexpr (complex<T>) {
        return typename T::value_type{};
    } else {
        return T{};
    }
}

}  // namespace detail

template<typename RealOrComplex>
using real_or_complex_value_t = decltype(detail::real_or_complex_value<RealOrComplex>());

template<typename FloatOrComplex, typename URNG = std::mt19937>
[[nodiscard]] auto generate_noise_signal(std::size_t length, typename URNG::result_type seed)
{
    using Float = real_or_complex_value_t<FloatOrComplex>;
#if defined(NEO_HAS_BUILTIN_FLOAT16)
    using StandardFloat = std::conditional_t<std::same_as<Float, _Float16>, float, Float>;
#else
    using StandardFloat = Float;
#endif

    auto rng    = URNG{seed};
    auto dist   = std::uniform_real_distribution<StandardFloat>{StandardFloat(-1), StandardFloat(1)};
    auto signal = stdex::mdarray<FloatOrComplex, stdex::dextents<size_t, 1>>{length};

    if constexpr (complex<FloatOrComplex>) {
        std::generate_n(signal.data(), signal.size(), [&] {
            return FloatOrComplex{static_cast<Float>(dist(rng)), static_cast<Float>(dist(rng))};
        });
    } else {
        std::generate_n(signal.data(), signal.size(), [&] { return static_cast<Float>(dist(rng)); });
    }

    return signal;
}

template<std::floating_point Float>
[[nodiscard]] auto generate_identity_impulse(std::size_t block_size, std::size_t num_subfilter)
    -> stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 2>>
{
    auto const num_bins = block_size + 1;
    auto impulse        = stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 2>>{num_subfilter, num_bins};
    fill(stdex::submdspan(impulse.to_mdspan(), 0, stdex::full_extent), std::complex{Float(1), Float(0)});
    return impulse;
}

}  // namespace neo
