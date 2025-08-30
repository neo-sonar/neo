#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdio>
#include <numbers>
#include <span>

template<std::integral T>
[[nodiscard]] constexpr auto ipow(T base, T exponent) -> T
{
    T result = 1;
    for (T i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}

constexpr auto bitrevorder(std::span<std::complex<double>> x) -> void
{
    std::size_t j = 0;
    for (std::size_t i = 0; i < x.size() - 1zu; ++i) {
        if (i < j) {
            std::swap(x[i], x[j]);
        }
        std::size_t k = x.size() / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

/// Reorder input using base-radix digit reversal permutation.
template<std::size_t Radix>
auto digitrevorder(std::span<std::complex<double>> x) noexcept -> void
{
    auto const len = x.size();

    auto j = 0zu;
    for (auto i = 0zu; i < len - 1zu; i++) {
        if (i < j) {
            std::swap(x[i], x[j]);
        }
        auto k = (Radix - 1zu) * len / Radix;
        while (k <= j) {
            j -= k;
            k /= Radix;
        }
        j += k / (Radix - 1zu);
    }
}

// Chapter 2.4.3: The Cooley-Tukey Radix-4 Algorithm
// Computational frameworks for the fast fourier transform
// ISBN: 978-0-89871-285-8
auto c2c_dit4(std::span<std::complex<double>> x, std::size_t order, double sign) -> void
{
    static constexpr auto Radix = 4zu;
    static constexpr auto TwoPi = 3.14159265359 * 2.0;

    auto const I = std::complex{0.0, sign};
    auto const t = order;
    auto const n = x.size();

    digitrevorder<Radix>(x);

    for (auto q{1zu}; q <= t; ++q) {
        auto const L  = ipow(Radix, q);
        auto const LR = L / Radix;
        auto const r  = n / L;

        for (auto j{0zu}; j < LR; ++j) {
            auto const angle = sign * TwoPi * static_cast<double>(j) / static_cast<double>(L);
            auto const w1    = std::polar(1.0, angle * 1.0);
            auto const w2    = std::polar(1.0, angle * 2.0);
            auto const w3    = std::polar(1.0, angle * 3.0);

            for (auto k{0zu}; k < r; ++k) {
                auto const a = x[k * L + LR * 0 + j];
                auto const b = x[k * L + LR * 1 + j] * w1;
                auto const c = x[k * L + LR * 2 + j] * w2;
                auto const d = x[k * L + LR * 3 + j] * w3;

                auto const t0 = a + c;
                auto const t1 = a - c;
                auto const t2 = b + d;
                auto const t3 = b - d;

                x[k * L + LR * 0 + j] = t0 + t2;
                x[k * L + LR * 1 + j] = t1 - t3 * I;
                x[k * L + LR * 2 + j] = t0 - t2;
                x[k * L + LR * 3 + j] = t1 + t3 * I;
            }
        }
    }
}

auto c2c_dit4(
    std::span<std::complex<double>> x,
    std::span<std::complex<double> const> w,
    std::size_t order,
    double sign
) -> void
{
    static constexpr auto Radix = 4zu;
    static constexpr auto TwoPi = 3.14159265359 * 2.0;

    auto const I = std::complex{0.0, sign};
    auto const t = order;
    auto const n = x.size();

    digitrevorder<Radix>(x);

    for (auto q{1zu}; q <= t; ++q) {
        auto const L  = ipow(Radix, q);
        auto const LR = L / Radix;
        auto const r  = n / L;

        for (auto j{0zu}; j < LR; ++j) {
            auto const w1 = w[j];
            auto const w2 = w1 * w1;
            auto const w3 = w2 * w1;

            for (auto k{0zu}; k < r; ++k) {
                auto const a = x[k * L + LR * 0 + j];
                auto const b = x[k * L + LR * 1 + j] * w1;
                auto const c = x[k * L + LR * 2 + j] * w2;
                auto const d = x[k * L + LR * 3 + j] * w3;

                auto const t0 = a + c;
                auto const t1 = a - c;
                auto const t2 = b + d;
                auto const t3 = b - d;

                x[k * L + LR * 0 + j] = t0 + t2;
                x[k * L + LR * 1 + j] = t1 - t3 * I;
                x[k * L + LR * 2 + j] = t0 - t2;
                x[k * L + LR * 3 + j] = t1 + t3 * I;
            }
        }
    }
}

auto c2c_stockham(
    std::span<std::complex<double>> x,
    std::span<std::complex<double>> work,
    std::size_t order,
    double sign
) -> void
{
    static constexpr auto Radix = 2zu;
    static constexpr auto TwoPi = 3.14159265359 * 2.0;

    auto const n = ipow(std::size_t(2), order);

    auto l = n / 2U;
    auto m = 1zu;

    for (auto t{1zu}; t <= order; ++t) {
        for (auto j{0zu}; j < l; ++j) {
            auto const angle = sign * TwoPi * static_cast<double>(j) / static_cast<double>(2 * l);
            auto const w     = std::polar(1.0, angle);
            for (auto k{0zu}; k < m; ++k) {
                auto const c0 = x[k + j * m];
                auto const c1 = x[k + j * m + l * m];

                work[k + 2 * j * m]     = c0 + c1;
                work[k + 2 * j * m + m] = (c0 - c1) * w;
            }
        }

        l = l / 2U;
        m = m * 2U;
        std::swap(x, work);
    }

    if ((order & 1zu) != 0) {
        std::ranges::copy(x, work.begin());
    }
}

auto main() -> int
{
    auto print = [](auto const* msg, auto x, auto scale) {
        std::printf("%s\n", msg);
        for (auto z : x) {
            std::printf("%+.4f,%+.4f\n", z.real() * scale, z.imag() * scale);
        }
        std::puts("");
    };

    static constexpr auto TwoPi = 3.14159265359 * 2.0;
    static constexpr auto Radix = 4;
    static constexpr auto Order = 2;
    static constexpr auto Size  = ipow(Radix, Order);

    auto w_fwd = std::array<std::complex<double>, Size / 4>{};
    auto w_bwd = std::array<std::complex<double>, Size / 4>{};
    for (auto i{0U}; i < w_fwd.size(); ++i) {
        auto angle = TwoPi * static_cast<double>(i) / static_cast<double>(Size);
        w_fwd[i]   = std::polar(1.0, -angle);
        w_bwd[i]   = std::polar(1.0, +angle);
    }

    auto x    = std::array<std::complex<double>, Size>{};
    auto work = std::array<std::complex<double>, Size>{};
    x[0]      = 1.0;
    x[1]      = 2.0;
    x[2]      = 3.0;
    print("input", x, 1.0);

    c2c_dit4(x, Order, -1.0);
    print("fwd", x, 1.0);

    c2c_dit4(x, Order, +1.0);
    print("bwd", x, 1.0 / double(Size));

    return 0;
}
