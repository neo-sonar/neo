// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/complex/split_complex.hpp>
#include <neo/container/csr_matrix.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/simd/native.hpp>

#if defined(NEO_HAS_APPLE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#endif

#if defined(NEO_HAS_XSIMD)
    #include <neo/algorithm/backend/xsimd.hpp>
#endif

#include <cassert>
#include <utility>

namespace neo::native {

namespace detail {

template<typename Batch>
auto multiply_add(
    typename Batch::float_type const* x_real,
    typename Batch::float_type const* x_imag,
    typename Batch::float_type const* y_real,
    typename Batch::float_type const* y_imag,
    typename Batch::float_type const* z_real,
    typename Batch::float_type const* z_imag,
    typename Batch::float_type* out_real,
    typename Batch::float_type* out_imag,
    std::size_t size
) -> void
{
    using reg = Batch;

    auto const inc      = reg::size;
    auto const vec_size = size - (size % inc);

    for (auto i = std::size_t(0); i < vec_size; i += inc) {
        auto const xre = reg::loadu(&x_real[i]);
        auto const xim = reg::loadu(&x_imag[i]);
        auto const yre = reg::loadu(&y_real[i]);
        auto const yim = reg::loadu(&y_imag[i]);
        auto const zre = reg::loadu(&z_real[i]);
        auto const zim = reg::loadu(&z_imag[i]);

        auto const out_re = reg::add(reg::sub(reg::mul(xre, yre), reg::mul(xim, yim)), zre);
        auto const out_im = reg::add(reg::add(reg::mul(xre, yim), reg::mul(xim, yre)), zim);

        reg::storeu(&out_real[i], out_re);
        reg::storeu(&out_imag[i], out_im);
    }

    for (auto i = vec_size; i < size; ++i) {
        auto const xre = x_real[i];
        auto const xim = x_imag[i];
        auto const yre = y_real[i];
        auto const yim = y_imag[i];

        out_real[i] = (xre * yre - xim * yim) + z_real[i];
        out_imag[i] = (xre * yim + xim * yre) + z_imag[i];
    }
}

}  // namespace detail

#if defined(NEO_HAS_APPLE_ACCELERATE)
    #define NEO_HAS_NATIVE_SPLIT_COMPLEX_MULTIPLY_ADD

template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
auto multiply_add(
    Float const* x_real,
    Float const* x_imag,
    Float const* y_real,
    Float const* y_imag,
    Float const* z_real,
    Float const* z_imag,
    Float* out_real,
    Float* out_imag,
    std::size_t size
) -> void
{
    using split_t = std::conditional_t<std::same_as<Float, float>, DSPSplitComplex, DSPDoubleSplitComplex>;

    auto x_sc = split_t{.realp = const_cast<Float*>(x_real), .imagp = const_cast<Float*>(x_imag)};
    auto y_sc = split_t{.realp = const_cast<Float*>(y_real), .imagp = const_cast<Float*>(y_imag)};
    auto z_sc = split_t{.realp = const_cast<Float*>(z_real), .imagp = const_cast<Float*>(z_imag)};
    auto o_sc = split_t{.realp = out_real, .imagp = out_imag};

    if constexpr (std::same_as<Float, float>) {
        vDSP_zvma(&x_sc, 1, &y_sc, 1, &z_sc, 1, &o_sc, 1, size);
    } else {
        vDSP_zvmaD(&x_sc, 1, &y_sc, 1, &z_sc, 1, &o_sc, 1, size);
    }
}

#elif defined(NEO_HAS_SIMD_SSE2) and not defined(NEO_HAS_SIMD_AVX)
    #define NEO_HAS_NATIVE_SPLIT_COMPLEX_MULTIPLY_ADD

struct batch_f32
{
    using float_type = float;

    static constexpr auto const size   = 128 / 32;
    static constexpr auto const loadu  = _mm_loadu_ps;
    static constexpr auto const storeu = _mm_storeu_ps;
    static constexpr auto const add    = _mm_add_ps;
    static constexpr auto const sub    = _mm_sub_ps;
    static constexpr auto const mul    = _mm_mul_ps;
};

struct batch_f64
{
    using float_type = double;

    static constexpr auto const size   = 128 / 64;
    static constexpr auto const loadu  = _mm_loadu_pd;
    static constexpr auto const storeu = _mm_storeu_pd;
    static constexpr auto const add    = _mm_add_pd;
    static constexpr auto const sub    = _mm_sub_pd;
    static constexpr auto const mul    = _mm_mul_pd;
};

template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
auto multiply_add(
    Float const* x_real,
    Float const* x_imag,
    Float const* y_real,
    Float const* y_imag,
    Float const* z_real,
    Float const* z_imag,
    Float* out_real,
    Float* out_imag,
    std::size_t size
) -> void
{
    using batch = std::conditional_t<std::same_as<Float, float>, batch_f32, batch_f64>;
    native::detail::multiply_add<batch>(x_real, x_imag, y_real, y_imag, z_real, z_imag, out_real, out_imag, size);
}

#elif defined(NEO_HAS_SIMD_AVX) and not defined(NEO_HAS_SIMD_AVX512F)
    #define NEO_HAS_NATIVE_SPLIT_COMPLEX_MULTIPLY_ADD

struct batch_f32
{
    using float_type = float;

    static constexpr auto const size   = 256 / 32;
    static constexpr auto const loadu  = _mm256_loadu_ps;
    static constexpr auto const storeu = _mm256_storeu_ps;
    static constexpr auto const add    = _mm256_add_ps;
    static constexpr auto const sub    = _mm256_sub_ps;
    static constexpr auto const mul    = _mm256_mul_ps;
};

struct batch_f64
{
    using float_type = double;

    static constexpr auto const size   = 256 / 64;
    static constexpr auto const loadu  = _mm256_loadu_pd;
    static constexpr auto const storeu = _mm256_storeu_pd;
    static constexpr auto const add    = _mm256_add_pd;
    static constexpr auto const sub    = _mm256_sub_pd;
    static constexpr auto const mul    = _mm256_mul_pd;
};

template<std::floating_point Float>
    requires(std::same_as<Float, float> or std::same_as<Float, double>)
auto multiply_add(
    Float const* x_real,
    Float const* x_imag,
    Float const* y_real,
    Float const* y_imag,
    Float const* z_real,
    Float const* z_imag,
    Float* out_real,
    Float* out_imag,
    std::size_t size
) -> void
{
    using batch = std::conditional_t<std::same_as<Float, float>, batch_f32, batch_f64>;
    native::detail::multiply_add<batch>(x_real, x_imag, y_real, y_imag, z_real, z_imag, out_real, out_imag, size);
}

#endif

}  // namespace neo::native

namespace neo {

// out = x * y + z
template<in_vector VecX, in_vector VecY, in_vector VecZ, out_vector VecOut>
constexpr auto multiply_add(VecX x, VecY y, VecZ z, VecOut out) noexcept -> void
{
    assert(detail::extents_equal(x, y, z, out));

#if defined(NEO_HAS_XSIMD)
    if constexpr (always_vectorizable<VecX, VecY, VecZ, VecOut>) {
        auto x_ptr   = x.data_handle();
        auto y_ptr   = y.data_handle();
        auto z_ptr   = z.data_handle();
        auto out_ptr = out.data_handle();
        auto size    = x.extent(0);
        if constexpr (requires { detail::multiply_add(x_ptr, y_ptr, z_ptr, out_ptr, size); }) {
            detail::multiply_add(x_ptr, y_ptr, z_ptr, out_ptr, size);
            return;
        }
    }
#endif

    for (decltype(x.extent(0)) i{0}; i < x.extent(0); ++i) {
        out[i] = x[i] * y[i] + z[i];
    }
}

// out = x * y + z
template<typename U, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_add(
    in_vector auto x,
    csr_matrix<U, IndexType, ValueContainer, IndexContainer> const& y,
    typename csr_matrix<U, IndexType, ValueContainer, IndexContainer>::index_type y_row,
    in_vector auto z,
    out_vector auto out
) noexcept -> void
{
    assert(x.extent(0) == y.columns());

    auto const& rrows = y.row_container();
    auto const& rcols = y.column_container();
    auto const& rvals = y.value_container();

    for (auto i{rrows[y_row]}; i < rrows[y_row + 1]; ++i) {
        auto col = rcols[i];
        out[col] = x[col] * rvals[i] + z[col];
    }
}

// out = x * y + z
template<in_vector VecX, in_vector VecY, in_vector VecZ, out_vector VecOut>
constexpr auto
multiply_add(split_complex<VecX> x, split_complex<VecY> y, split_complex<VecZ> z, split_complex<VecOut> out) noexcept
    -> void
{
    assert(detail::extents_equal(x.real, x.imag, y.real, y.imag, z.real, z.imag, out.real, out.imag));

    constexpr auto const same_type    = detail::all_same_value_type_v<VecX, VecY, VecZ, VecOut>;
    constexpr auto const vectorizable = same_type and always_vectorizable<VecX, VecY, VecZ, VecOut>;

    if constexpr (vectorizable) {
        [[maybe_unused]] auto const size = static_cast<size_t>(x.real.extent(0));

        [[maybe_unused]] auto const* xre = x.real.data_handle();
        [[maybe_unused]] auto const* xim = x.imag.data_handle();
        [[maybe_unused]] auto const* yre = y.real.data_handle();
        [[maybe_unused]] auto const* yim = y.imag.data_handle();
        [[maybe_unused]] auto const* zre = z.real.data_handle();
        [[maybe_unused]] auto const* zim = z.imag.data_handle();

        [[maybe_unused]] auto* ore = out.real.data_handle();
        [[maybe_unused]] auto* oim = out.imag.data_handle();

#if defined(NEO_HAS_NATIVE_SPLIT_COMPLEX_MULTIPLY_ADD)
        if constexpr (requires { native::multiply_add(xre, xim, yre, yim, zre, zim, ore, oim, size); }) {
            native::multiply_add(xre, xim, yre, yim, zre, zim, ore, oim, size);
            return;
        }
#elif defined(NEO_HAS_XSIMD)
        if constexpr (requires { detail::multiply_add(xre, xim, yre, yim, zre, zim, ore, oim, size); }) {
            detail::multiply_add(xre, xim, yre, yim, zre, zim, ore, oim, size);
            return;
        }
#endif
    }

    for (auto i{0}; i < static_cast<int>(x.real.extent(0)); ++i) {
        auto const xre = x.real[i];
        auto const xim = x.imag[i];
        auto const yre = y.real[i];
        auto const yim = y.imag[i];

        out.real[i] = (xre * yre - xim * yim) + z.real[i];
        out.imag[i] = (xre * yim + xim * yre) + z.imag[i];
    }
}

}  // namespace neo
