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
    for (std::size_t i = 0; i < x.size() - 1U; ++i) {
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

    auto j = 0UL;
    for (auto i = 0UL; i < len - 1UL; i++) {
        if (i < j) {
            std::swap(x[i], x[j]);
        }
        auto k = (Radix - 1UL) * len / Radix;
        while (k <= j) {
            j -= k;
            k /= Radix;
        }
        j += k / (Radix - 1);
    }
}

// Chapter 2.4.3: The Cooley-Tukey Radix-4 Algorithm
// Computational frameworks for the fast fourier transform
// ISBN: 978-0-89871-285-8
auto c2c_dit4(std::span<std::complex<double>> x, std::size_t order, double sign) -> void
{
    static constexpr auto Radix = 4UL;
    static constexpr auto TwoPi = 3.14159265359 * 2.0;

    auto const I = std::complex{0.0, sign};
    auto const t = order;
    auto const n = x.size();

    digitrevorder<Radix>(x);

    for (auto q{1UL}; q <= t; ++q) {
        auto const L  = ipow(Radix, q);
        auto const LR = L / Radix;
        auto const r  = n / L;

        for (auto j{0UL}; j < LR; ++j) {
            auto const angle = sign * TwoPi * static_cast<double>(j) / static_cast<double>(L);
            auto const w1    = std::polar(1.0, angle * 1.0);
            auto const w2    = std::polar(1.0, angle * 2.0);
            auto const w3    = std::polar(1.0, angle * 3.0);

            for (auto k{0UL}; k < r; ++k) {
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

auto main() -> int
{
    auto print = [](auto const* msg, auto x, auto scale) {
        std::printf("%s\n", msg);
        for (auto z : x) {
            std::printf("%+.4f,%+.4f\n", z.real() * scale, z.imag() * scale);
        }
        std::puts("");
    };

    static constexpr auto Order = 2;
    static constexpr auto Size  = ipow(4, Order);

    auto x = std::array<std::complex<double>, Size>{};
    x[0]   = 1.0;
    x[1]   = 2.0;
    x[2]   = 3.0;
    print("input", x, 1.0);

    c2c_dit4(x, Order, -1.0);
    print("fwd", x, 1.0);

    c2c_dit4(x, Order, +1.0);
    print("bwd", x, 1.0 / double(Size));

    return 0;
}
