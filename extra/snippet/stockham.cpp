#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

struct c2c_stockham_dif2_plan
{
    explicit c2c_stockham_dif2_plan(std::size_t size) : _w(make_twiddle_lut(size)), _work(size) {}

    auto size() const noexcept { return _work.size(); }

    // Fourier transform
    // x : input/output sequence
    auto fft(std::complex<double>* x) -> void { fft0(size(), 1, 0, x, _work.data(), _w.data()); }

    // Inverse Fourier transform
    // x : input/output sequence
    void ifft(std::complex<double>* x)
    {
        for (std::size_t p = 0; p < size(); p++) {
            x[p] = std::conj(x[p]);
        }
        fft0(size(), 1, 0, x, _work.data(), _w.data());
        for (std::size_t k = 0; k < size(); k++) {
            x[k] = std::conj(x[k]);
        }
    }

private:
    // n  : sequence length
    // s  : stride
    // eo : x is output if eo == 0, work is output if eo == 1
    // x  : input sequence(or output sequence if eo == 0)
    // work  : work area(or output sequence if eo == 1)
    static auto fft0(
        std::size_t n,
        std::size_t s,
        bool eo,
        std::complex<double>* x,
        std::complex<double>* work,
        std::complex<double> const* w
    ) -> void
    {
        if (n == 1) {
            if (eo) {
                for (std::size_t q = 0; q < s; q++) {
                    work[q] = x[q];
                }
            }
            return;
        }

        auto const m = n / 2zu;

        for (std::size_t p = 0; p < m; p++) {
            auto wp = w[p * s];

            for (std::size_t q = 0; q < s; q++) {
                auto const a = x[q + s * (p + 0)];
                auto const b = x[q + s * (p + m)];

                work[q + s * (2 * p + 0)] = a + b;
                work[q + s * (2 * p + 1)] = (a - b) * wp;
            }
        }

        fft0(n / 2, 2 * s, !eo, work, x, w);
    }

    static auto make_twiddle_lut(size_t n) -> std::vector<std::complex<double>>
    {
        auto const theta0 = 2.0 * M_PI / static_cast<double>(n);

        auto w = std::vector<std::complex<double>>(n / 2);
        for (std::size_t i = 0; i < w.size(); i++) {
            w[i] = std::polar(1.0, i * theta0);
        }
        return w;
    }

    std::vector<std::complex<double>> _w;
    std::vector<std::complex<double>> _work;
};

auto print = [](auto const* msg, auto x, auto scale) {
    std::puts(msg);
    for (auto z : x) {
        std::printf("%+.4f,%+.4f\n", z.real() * scale, z.imag() * scale);
    }
    std::puts("");
};

auto main() -> int
{
    static constexpr auto const Order = 2zu;
    static constexpr auto const Size  = 1zu << Order;

    auto plan = c2c_stockham_dif2_plan{Size};
    auto x    = std::array<std::complex<double>, Size>{};
    x[0]      = 1.0;
    x[1]      = 2.0;
    x[2]      = 3.0;
    print("input", x, 1.0);

    plan.fft(x.data());
    print("fwd", x, 1.0);

    plan.ifft(x.data());
    print("bwd", x, 1.0 / double(Size));

    return 0;
}
