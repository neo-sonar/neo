// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/container/mdspan.hpp>

#if defined(NEO_HAS_APPLE_ACCELERATE)
    #if __MAC_OS_X_VERSION_MIN_REQUIRED >= 130300
        #include <Accelerate/Accelerate.h>
        #define NEO_HAS_CBLAS 1
    #endif
#elif defined(NEO_HAS_INTEL_MKL)
    #include <mkl.h>
    #define NEO_HAS_CBLAS 1
#endif

#include <complex>
#include <concepts>

#if defined(NEO_HAS_CBLAS)
namespace neo::cblas {

    #if defined(NEO_HAS_APPLE_ACCELERATE)
using size_type = long;
    #elif defined(NEO_HAS_INTEL_MKL)
using size_type = MKL_INT;
    #endif

auto scal(std::integral auto n, float alpha, float* x, std::integral auto inc_x) noexcept
{
    cblas_sscal(static_cast<size_type>(n), alpha, x, static_cast<size_type>(inc_x));
}

auto scal(std::integral auto n, double alpha, double* x, std::integral auto inc_x) noexcept
{
    cblas_dscal(static_cast<size_type>(n), alpha, x, static_cast<size_type>(inc_x));
}

auto scal(std::integral auto n, std::complex<float> alpha, std::complex<float>* x, std::integral auto inc_x) noexcept

{
    cblas_cscal(static_cast<size_type>(n), &alpha, x, static_cast<size_type>(inc_x));
}

auto scal(std::integral auto n, std::complex<double> alpha, std::complex<double>* x, std::integral auto inc_x) noexcept

{
    cblas_zscal(static_cast<size_type>(n), &alpha, x, static_cast<size_type>(inc_x));
}

auto scal(std::integral auto n, float alpha, std::complex<float>* x, std::integral auto inc_x) noexcept
{
    cblas_csscal(static_cast<size_type>(n), alpha, x, static_cast<size_type>(inc_x));
}

auto scal(std::integral auto n, double alpha, std::complex<double>* x, std::integral auto inc_x) noexcept
{
    cblas_zdscal(static_cast<size_type>(n), alpha, x, static_cast<size_type>(inc_x));
}

}  // namespace neo::cblas

#endif
